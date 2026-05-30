#!/usr/bin/env python
import tilelang
import tilelang.language as T
import torch

from ....common.spec import DispatchField, TilelangKernel, register_kernel

symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")

DIM_PER_CORE = 2048

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_decode_kernel_cache = {}


def build_causal_conv1d_decode_kernel(
    width: int,
    dim_chunks: int,
    num_batches: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "float16",
    has_silu: bool = True,
) -> torch.nn.Module:
    hist_len = width - 1
    symbol_dim = T.symbolic("dim")
    symbol_batch = T.symbolic("batch_size")
    total_tasks = num_batches * dim_chunks

    @T.prim_func
    def causal_conv1d_decode(
        x: T.Tensor((symbol_batch, symbol_dim), dtype_str),
        weight: T.Tensor((width, symbol_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, symbol_dim), dtype_str),
        conv_state_indices_init: T.Tensor((num_batches,), "int32"),
        conv_state_indices_current: T.Tensor((num_batches,), "int32"),
        initial_state_mode: T.Tensor((num_batches,), "int32"),
        bias: T.Tensor((symbol_dim,), dtype_str),
        y: T.Tensor((symbol_batch, symbol_dim), dtype_str),
    ):
        with T.Kernel(total_tasks, is_npu=True) as (cid, vid):
            batch_id = cid // dim_chunks
            dim_chunk_id = cid % dim_chunks

            d_offset = dim_chunk_id * dim_per_core
            read_cache_line = conv_state_indices_init[batch_id]
            write_cache_line = conv_state_indices_current[batch_id]
            has_initial = initial_state_mode[batch_id]

            hist0 = T.alloc_ub((dim_per_core,), dtype_str)
            hist1 = T.alloc_ub((dim_per_core,), dtype_str)
            hist2 = T.alloc_ub((dim_per_core,), dtype_str)
            w0 = T.alloc_ub((dim_per_core,), dtype_str)
            w1 = T.alloc_ub((dim_per_core,), dtype_str)
            w2 = T.alloc_ub((dim_per_core,), dtype_str)
            w3 = T.alloc_ub((dim_per_core,), dtype_str)
            bias_ub = T.alloc_ub((dim_per_core,), dtype_str)
            x_ub = T.alloc_ub((dim_per_core,), dtype_str)
            state0 = T.alloc_ub((dim_per_core,), dtype_str)
            tmp = T.alloc_ub((dim_per_core,), dtype_str)
            y_ub = T.alloc_ub((dim_per_core,), dtype_str)
            save0 = T.alloc_ub((dim_per_core,), dtype_str)
            save1 = T.alloc_ub((dim_per_core,), dtype_str)
            save2 = T.alloc_ub((dim_per_core,), dtype_str)

            T.copy(weight[0, d_offset], w0)
            T.copy(weight[1, d_offset], w1)
            T.copy(weight[2, d_offset], w2)
            T.copy(weight[3, d_offset], w3)
            T.copy(bias[d_offset], bias_ub)
            T.barrier_all()

            T.tile.fill(hist0, 0.0)
            T.tile.fill(hist1, 0.0)
            T.tile.fill(hist2, 0.0)

            if has_initial != 0:
                if hist_len >= 1 and symbol_state_len > 0:
                    T.copy(conv_state[read_cache_line, 0, d_offset], hist0)
                if hist_len >= 2 and symbol_state_len > 1:
                    T.copy(conv_state[read_cache_line, 1, d_offset], hist1)
                if hist_len >= 3 and symbol_state_len > 2:
                    T.copy(conv_state[read_cache_line, 2, d_offset], hist2)

            T.copy(x[batch_id, d_offset], x_ub)
            T.barrier_all()

            T.tile.mul(state0, w0, hist0)
            T.tile.mul(tmp, w1, hist1)
            T.tile.add(state0, state0, tmp)
            T.tile.mul(tmp, w2, hist2)
            T.tile.add(state0, state0, tmp)
            T.tile.mul_add_dst(state0, x_ub, w3)
            T.tile.add(tmp, state0, bias_ub)
            if has_silu:
                T.tile.silu(y_ub, tmp)
            else:
                T.tile.add(y_ub, state0, bias_ub)

            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(y_ub, y[batch_id, d_offset])

            T.copy(hist1, save0)
            T.copy(hist2, save1)
            T.copy(x_ub, save2)

            T.barrier_all()
            if hist_len >= 1 and symbol_state_len > 0:
                T.copy(save0, conv_state[write_cache_line, 0, d_offset])
            if hist_len >= 2 and symbol_state_len > 1:
                T.copy(save1, conv_state[write_cache_line, 1, d_offset])
            if hist_len >= 3 and symbol_state_len > 2:
                T.copy(save2, conv_state[write_cache_line, 2, d_offset])

    return causal_conv1d_decode


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config)
def _build_decode_kernel_jit(
    width: int,
    dim_chunks: int,
    num_batches: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "float16",
    has_silu: bool = True,
) -> torch.nn.Module:
    return build_causal_conv1d_decode_kernel(
        width=width,
        dim_chunks=dim_chunks,
        num_batches=num_batches,
        dim_per_core=dim_per_core,
        dtype_str=dtype_str,
        has_silu=has_silu,
    )


@register_kernel
class CausalConv1dDecodeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("batch_size", "int32"),
        DispatchField("dim", "int32"),
        DispatchField("width", "int32"),
        DispatchField("has_silu", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "bs1_d2048_w4_silu1_f16",
            "batch_size": 1,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d2048_w4_silu1_f16",
            "batch_size": 2,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs4_d2048_w4_silu1_f16",
            "batch_size": 4,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs8_d2048_w4_silu1_f16",
            "batch_size": 8,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs16_d2048_w4_silu1_f16",
            "batch_size": 16,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs32_d2048_w4_silu1_f16",
            "batch_size": 32,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs48_d2048_w4_silu1_f16",
            "batch_size": 48,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d4096_w4_silu1_f16",
            "batch_size": 1,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d4096_w4_silu1_f16",
            "batch_size": 2,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs4_d4096_w4_silu1_f16",
            "batch_size": 4,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs8_d4096_w4_silu1_f16",
            "batch_size": 8,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs16_d4096_w4_silu1_f16",
            "batch_size": 16,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs32_d4096_w4_silu1_f16",
            "batch_size": 32,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs48_d4096_w4_silu1_f16",
            "batch_size": 48,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d8192_w4_silu1_f16",
            "batch_size": 1,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d8192_w4_silu1_f16",
            "batch_size": 2,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs4_d8192_w4_silu1_f16",
            "batch_size": 4,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs8_d8192_w4_silu1_f16",
            "batch_size": 8,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs16_d8192_w4_silu1_f16",
            "batch_size": 16,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs32_d8192_w4_silu1_f16",
            "batch_size": 32,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs48_d8192_w4_silu1_f16",
            "batch_size": 48,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
    ]

    @staticmethod
    def generate_source(
        batch_size: int,
        dim: int,
        width: int,
        has_silu: int,
        dtype: str,
    ) -> str:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(
                f"CausalConv1D Decode TileLang kernel only supports "
                f"dtype=float16/bfloat16, got {dtype}"
            )
        dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_decode_kernel(
            width=width,
            dim_chunks=dim_chunks,
            num_batches=batch_size,
            dim_per_core=DIM_PER_CORE,
            dtype_str=dtype,
            has_silu=bool(has_silu),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config={
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_auto_sync": False,
                "tl.ascend_memory_planning": True,
            },
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def get_decode_kernel(
    width: int,
    num_batches: int,
    dim: int,
    dtype_str: str = "float16",
    has_silu: bool = True,
) -> torch.nn.Module:
    dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
    cache_key = (
        width,
        dim_chunks,
        num_batches,
        DIM_PER_CORE,
        dtype_str,
        has_silu,
    )
    if cache_key not in _decode_kernel_cache:
        _decode_kernel_cache[cache_key] = _build_decode_kernel_jit(
            width,
            dim_chunks,
            num_batches,
            DIM_PER_CORE,
            dtype_str,
            has_silu,
        )
    return _decode_kernel_cache[cache_key]


def causal_conv1d_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    original_dtype = x.dtype
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    has_silu = activation in ("silu", "swish")
    width = weight.shape[1]
    dim = weight.shape[0]

    if original_dtype == torch.bfloat16:
        x = x.to(torch.float16)
        weight = weight.to(torch.float16)
        conv_state_work = conv_state.to(torch.float16).clone()
        if bias is not None:
            bias_work = bias.to(torch.float16).contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.float16, device=conv_state.device)
    else:
        conv_state_work = conv_state.clone()
        if bias is not None:
            bias_work = bias.contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=conv_state.dtype, device=conv_state.device)

    weight_t = weight.transpose(0, 1).contiguous()
    conv_state_t = conv_state_work.transpose(1, 2).contiguous()

    if query_start_loc is not None:
        qsl_kernel = query_start_loc.to(torch.int32).contiguous()
        batch = qsl_kernel.numel() - 1
        x_kernel = x.contiguous()
    else:
        if x.dim() == 2:
            x_work = x.unsqueeze(-1)
        else:
            x_work = x
        batch, dim_check, seqlen = x_work.shape
        assert dim_check == dim
        assert seqlen == 1
        x_kernel = x_work.reshape(batch, dim).contiguous()

    if conv_state_indices is None:
        init_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
        current_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
    elif conv_state_indices.dim() == 1:
        ci = conv_state_indices.to(torch.int32).contiguous()
        init_indices = ci
        current_indices = ci.clone()
    else:
        ci = conv_state_indices.to(torch.int32).contiguous()
        if initial_state_idx is None:
            init_indices = ci[:, 0].contiguous()
        else:
            isi = initial_state_idx.to(torch.int32).contiguous()
            init_indices = torch.where(isi == 0, ci[:, 0], ci[:, 1]).contiguous()
        if block_idx_last_scheduled_token is None:
            current_indices = ci[:, 0].contiguous()
        else:
            bilt = block_idx_last_scheduled_token.to(torch.int32).contiguous()
            current_indices = torch.where(bilt == 0, ci[:, 0], ci[:, 1]).contiguous()

    initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)

    kernel_dtype_str = (
        "float16"
        if (original_dtype == torch.float16 or original_dtype == torch.bfloat16)
        else "float32"
    )

    kernel = get_decode_kernel(
        width, batch, dim, kernel_dtype_str, has_silu
    )
    output = kernel(
        x_kernel,
        weight_t,
        conv_state_t,
        init_indices,
        current_indices,
        initial_state_mode,
        bias_work,
    )

    conv_state.copy_(
        conv_state_t.transpose(1, 2).contiguous().to(original_dtype)
    )

    if query_start_loc is None and x.dim() == 2:
        output = output.squeeze(-1) if output.dim() == 3 and output.shape[-1] == 1 else output

    if original_dtype == torch.bfloat16:
        output = output.to(torch.bfloat16)

    return output