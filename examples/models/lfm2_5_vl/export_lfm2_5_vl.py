# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL as a single multi-method PTE for ExecuTorch with CUDA backend.

All three methods (vision encoder, token embedding, text decoder) are delegated
to the CUDA/AOTI backend.  Conv layer state is passed as explicit IO rather
than mutable buffers, which AOTI cannot re-trace.

Supports both checkpoint sizes:
  - LiquidAI/LFM2-VL-1.6B  (text dim 2048)
  - LiquidAI/LFM2.5-VL-450M (text dim 1024)

Methods (D = text hidden dim):
  vision_encoder  : [1, 3, 512, 512] f32 -> [1, 256, D] f32
  token_embedding : [1, seq_len] i64     -> [1, seq_len, D] f32
  text_decoder    : ([1, seq_len, D], [seq_len] i64) -> [1, 65536] f32

Usage:
    python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \\
        --model_dir LiquidAI/LFM2.5-VL-450M --dtype bf16
"""

import logging
import os
from argparse import ArgumentParser
from typing import Optional

import torch
from torch.export import Dim
from torch.nn.attention import SDPBackend

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.examples.models.lfm2_5_vl.model import (
    Lfm2p5VlModel,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
)

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Workaround: torch._inductor maps arch 103 (Blackwell B300) to "100f",
# but Triton generates PTX targeting sm_103a. The mismatch causes nvcc to
# fail with "SM version specified by .target is higher than default SM
# version assumed". Patch the mapping so nvcc -gencode matches the PTX.
from torch._inductor.codecache import cuda_compile_utils

_orig_nvcc_arch = cuda_compile_utils._nvcc_arch_as_compile_option


def _patched_nvcc_arch() -> str:
    arch = cuda_compile_utils.cuda_env.get_cuda_arch()
    if arch == "103":
        return "103a"
    return _orig_nvcc_arch()


cuda_compile_utils._nvcc_arch_as_compile_option = _patched_nvcc_arch

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _resolve_params_path(model_dir: str, params: Optional[str]) -> Optional[str]:
    """Pick a bundled config based on model_dir if --params was not provided."""
    if params is not None:
        return params
    name = model_dir.lower()
    if "450m" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_450m_config.json")
    if "1.6b" in name or "1_6b" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_1_6b_config.json")
    return None


# ---------------------------------------------------------------------------
# Per-method export helpers
# ---------------------------------------------------------------------------


def export_image_encoder(lfm2) -> torch.export.ExportedProgram:
    """Export vision encoder: [1,3,512,512] f32 pixels [0,255] -> [1,256,D] f32."""

    class ImageEncoder(torch.nn.Module):
        def __init__(self, lfm2):
            super().__init__()
            self.lfm2 = lfm2

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.lfm2.image_embedding(images)

    encoder = ImageEncoder(lfm2)
    example_pixels = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32
    )

    logging.info("Exporting vision encoder...")
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        return torch.export.export(encoder, (example_pixels,), strict=False)


def export_text_decoder(
    lfm2, dtype: torch.dtype = torch.bfloat16
) -> torch.export.ExportedProgram:
    """Export hybrid LFM2.5 decoder: (embeddings, input_pos) -> logits.

    Conv states are initialised as zeros inside the forward pass and
    threaded through layers via attn_options["conv_states"].  This avoids
    register_buffer mutations that AOTI cannot re-trace.
    """
    from executorch.examples.models.lfm2.short_conv import ShortConvBlock

    conv_layer_indices = [
        i for i, layer in enumerate(lfm2.text_model.layers)
        if isinstance(layer, ShortConvBlock)
    ]

    class TextDecoder(torch.nn.Module):
        def __init__(self, text_model, conv_dim, conv_L_cache, conv_indices):
            super().__init__()
            self.text_model = text_model
            self.conv_dim = conv_dim
            self.conv_L_cache = conv_L_cache
            self.conv_indices = conv_indices

        def forward(
            self, embeddings: torch.Tensor, input_pos: torch.Tensor
        ) -> torch.Tensor:
            conv_states = {
                idx: torch.zeros(
                    1, self.conv_dim, self.conv_L_cache - 1,
                    dtype=embeddings.dtype, device=embeddings.device,
                )
                for idx in self.conv_indices
            }
            attn_options = {"conv_states": conv_states}
            if self.text_model.use_kv_cache:
                attn_options["input_pos"] = input_pos
            return self.text_model(None, attn_options, embeddings)

    decoder = TextDecoder(
        lfm2.text_model,
        conv_dim=lfm2.text_model_args.dim,
        conv_L_cache=3,
        conv_indices=conv_layer_indices,
    )
    dim = lfm2.text_model_args.dim
    dummy_seq = 8
    dummy_embeddings = torch.randn(1, dummy_seq, dim, dtype=dtype)
    dummy_input_pos = torch.arange(dummy_seq, dtype=torch.int64)

    logging.info("Exporting text decoder...")
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        return torch.export.export(
            decoder,
            (dummy_embeddings, dummy_input_pos),
            strict=False,
        )


def export_token_embedding(lfm2) -> torch.export.ExportedProgram:
    """Export token embedding table: [1, seq_len] i64 -> [1, seq_len, D] f32."""
    embed_module = lfm2.model_.model.language_model.get_input_embeddings()
    token_dim = Dim("token_dim_1", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = [{1: token_dim}]
    example_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)

    logging.info("Exporting token embedding...")
    with torch.no_grad():
        return torch.export.export(
            embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=False
        )


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------


def export_all(
    model_dir: str,
    output: Optional[str],
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = MAX_SEQ_LEN,
    max_context_len: int = MAX_SEQ_LEN,
    params_path: Optional[str] = None,
    _return_program: bool = False,
):
    logging.info(f"Loading {model_dir}...")
    lfm2_model = Lfm2p5VlModel(
        model_dir=model_dir,
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        params_path=params_path,
        # Disable XNNPack-specific source transforms and KV cache.  The native
        # KVCache + custom SDPA use mutable register_buffer state that creates
        # unbacked symbols AOTI cannot re-trace.  For CUDA we rely on AOTI's
        # own SDPA kernels; KV caching is not yet supported on this path.
        use_sdpa_with_kv_cache_op=False,
        use_kv_cache=False,
    )
    # Export on CPU — the emitter reads tensor bytes via ctypes pointer,
    # which segfaults if the storage lives on CUDA.
    lfm2 = lfm2_model.get_eager_model().to(dtype=dtype)

    logging.info("[1/3] Exporting vision encoder...")
    vision_ep = export_image_encoder(lfm2)

    logging.info("[2/3] Exporting text decoder...")
    decoder_ep = export_text_decoder(lfm2, dtype=dtype)

    logging.info("[3/3] Exporting token embedding...")
    token_ep = export_token_embedding(lfm2)

    exported_programs = {
        "vision_encoder": vision_ep,
        "token_embedding": token_ep,
        "text_decoder": decoder_ep,
    }

    partitioners = {}
    for key in exported_programs:
        compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
        partitioners[key] = [CudaPartitioner(compile_specs)]

    metadata = {
        "get_max_seq_len": lfm2.text_model_args.max_seq_len,
        "get_max_context_len": lfm2.text_model_args.max_context_len,
        "get_n_layers": lfm2.text_model_args.n_layers,
        "get_vocab_size": lfm2.text_model_args.vocab_size,
        "use_kv_cache": lfm2.text_model_args.use_kv_cache,
        "use_sdpa_with_kv_cache": lfm2.text_model_args.use_sdpa_with_kv_cache_op,
        "enable_dynamic_shape": lfm2.text_model_args.enable_dynamic_shape,
        "get_eos_ids": [7],
    }

    logging.info("Lowering to Edge IR...")
    et_prog = to_edge_transform_and_lower(
        exported_programs,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    logging.info("Finalizing ExecuTorch program...")
    et_program = et_prog.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    for plan in et_program._emitter_output.program.execution_plan:
        logging.info(f"Activation memory: {plan.non_const_buffer_sizes}")

    if _return_program:
        return et_program

    logging.info(f"Saving {output}...")
    with open(output, "wb") as f:
        et_program.write_to_file(f)
    logging.info(f"Saved {output} — methods: {et_program.methods}")


def main():
    parser = ArgumentParser(description="Export LFM2.5-VL to ExecuTorch")
    parser.add_argument(
        "--model_dir",
        default="LiquidAI/LFM2.5-VL-450M",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype (default: bf16)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Path to model params JSON (architecture config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: lfm2_5_vl_<dtype>_cuda.pte)",
    )
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    params_path = _resolve_params_path(args.model_dir, args.params)
    output = args.output or f"lfm2_5_vl_{args.dtype}_cuda.pte"

    export_all(
        args.model_dir,
        output,
        dtype,
        args.max_seq_len,
        args.max_seq_len,
        params_path,
    )


if __name__ == "__main__":
    main()
