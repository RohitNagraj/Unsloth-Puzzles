import triton.language as tl
import triton
import torch
import os

# Uncomment the following lines for debug
# os.environ['TRITON_DEBUG'] = '1'
# os.environ['TRITON_INTERPRET'] = '1'
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'


major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining the table outside the caller function since the caller function is benchmarked and this would add to it.
nf4_table = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                          -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                          0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                          0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=torch.float32,
                         device=device)


def my_dequantize(weight):
    out = _my_dequantize(weight.weight, weight.weight.quant_state)
    return out


def _my_dequantize(W, quant_state):
    absmax = quant_state.absmax
    shape = quant_state.shape
    dtype = quant_state.dtype
    blocksize_nf4 = quant_state.blocksize
    offset = quant_state.offset
    state2 = quant_state.state2
    absmax2 = state2.absmax
    uint8_lookup = state2.code
    blocksize_uint8 = state2.blocksize

    n_absmax = absmax.numel()
    out = torch.empty(shape, dtype=dtype, device="cuda:0", requires_grad=False)

    my_out = my_fused_dequantize(W, absmax, uint8_lookup, absmax2, blocksize_uint8,
                                 blocksize_nf4, n_absmax,
                                 out.numel(), offset, shape, dtype)

    return my_out


TILE_SIZE = 2048


# ------------------------------- FUSED KERNEL ------------------------------------------

@triton.autotune(configs=[triton.Config(kwargs={'TILE_SIZE': T}, num_warps=NW, num_stages=NS)
                          for T in [2048, 4096]
                          for NW in [2, 4, 8]
                          for NS in [2, 3, 4, 7]],
                 key=['is_bf16'],
                 use_cuda_graph=True,
                 )
@triton.jit
def combined_dequant_kernel(
        W_ptr,
        absmax_quantized_ptr,
        absmax2_ptr,
        out_ptr,
        uint8_lookup: tl.tensor,
        nf4_table: tl.tensor,
        offset: tl.tensor,  # 0.2...
        blocksize_uint8: tl.constexpr,  # 256
        blocksize_nf4: tl.constexpr,  # 64
        is_bf16: tl.constexpr,
        TILE_SIZE: tl.constexpr,  # 2048
):
    # -------------------------------------------------
    # Stage 1: Dequantize the UINT8–quantized absmax.
    # -------------------------------------------------

    pid = tl.program_id(0)

    W_start = pid * TILE_SIZE
    W_offsets = W_start + tl.arange(0, TILE_SIZE)  # 2048
    out_start = pid * TILE_SIZE * 2
    out_idx = out_start + tl.arange(0, 2 * TILE_SIZE)  # 4096

    absmax_start = W_start // (blocksize_nf4 // 2)
    absmax_offsets = absmax_start + \
        tl.arange(0, (TILE_SIZE // (blocksize_nf4 // 2)))
    absmax_quantized = tl.load(absmax_quantized_ptr + absmax_offsets, eviction_policy='evict_first').cast(
        tl.int32)  # 2048 // (64//2) = 64

    absmax2_start = W_start // (blocksize_uint8 * blocksize_nf4 // 2)
    absmax2_offsets = absmax2_start + \
        (tl.arange(0, 1 + (TILE_SIZE // (blocksize_uint8 * blocksize_nf4 // 2))))
    absmax2 = tl.load(absmax2_ptr + absmax2_offsets,
                      eviction_policy='evict_last')  # 1+ (64 // 256) = 1

    absmax = (tl.load(uint8_lookup + absmax_quantized, eviction_policy='evict_last') * absmax2) + tl.load(offset,
                                                                                                          eviction_policy='evict_last')  # 64 fp32

    # -------------------------------------------------
    # Stage 2: Dequantize the NF4–quantized weights.
    # -------------------------------------------------

    W = tl.load(W_ptr + W_offsets, eviction_policy='evict_first')  # 2048
    out_quantized = tl.interleave(W >> 4, W & 0x0F)  # 2 * TILE_SIZE = 4096

    nf4 = tl.load(nf4_table + out_quantized,
                  eviction_policy='evict_last')  # 2 * TILE_SIZE = 4096

    nf4 = nf4.reshape((TILE_SIZE // (blocksize_nf4 // 2)),
                      # (64 x 32).T = (32 x 64)
                      2 * TILE_SIZE // (TILE_SIZE // (blocksize_nf4 // 2))).trans()
    result = nf4 * absmax  # (32 x 64) * (64) = (32 x 64)

    # (32x64).T = (64x32).reshape = (2048)
    result = result.trans().reshape(2 * TILE_SIZE)

    if is_bf16:
        tl.store(out_ptr + out_idx, tl.cast(result, tl.bfloat16),
                 eviction_policy='evict_first')
    else:
        tl.store(out_ptr + out_idx, tl.cast(result, tl.float16),
                 eviction_policy='evict_first')


def my_fused_dequantize(W, absmax_quantized, uint8_lookup, absmax2, blocksize_uint8, blocksize_nf4, n_absmax, n_out,
                        offset, shape, dtype):
    """

    :param W: 8M x 1,uint8
    :param absmax_quantized: 256k x 1,uint8
    :param uint8_lookup: 256 x 1, float32
    :param absmax2: 1024 x 1, float32
    :param blocksize_uint8: 256
    :param blocksize_nf4: 64
    :param n_absmax: 256k
    :param n_out: 16M
    :param offset: float
    :param shape: [8192, 2048]
    :return:
    """

    # I'll launch W.numel() / TILE_SIZE number of blocks.
    grid = (W.numel() // TILE_SIZE,)
    # Each block in the grid has to process TILE_SIZE elements in W
    # If TILE_SIZE is < 8192, all the absmax will have one value of absmax2. Else, there will be higher number of absmax2

    out = torch.empty(n_out, dtype=dtype, device=W.device)
    is_bf16 = (dtype == torch.bfloat16)

    combined_dequant_kernel[grid](
        W,
        absmax_quantized,
        absmax2,
        out,
        uint8_lookup,
        nf4_table,
        offset,  # constant offset (Python scalar)
        blocksize_uint8,  # number of absmax values computed per block when quantizing to unit8
        blocksize_nf4,
        is_bf16,
        TILE_SIZE,
    )

    out = out.view(shape)
    return out
