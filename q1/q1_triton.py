import os

# os.environ['TRITON_DEBUG'] = '1'
# os.environ['TRITON_INTERPRET'] = '1'

import torch
import ctypes

import triton
import triton.language as tl

ctypes_c_int = ctypes.c_int

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def my_dequantize(weight):
    out = _my_dequantize_bf16(weight.weight, weight.weight.quant_state)
    # try:
    #     assert (out == my_out).sum().item() == out.numel()
    # except:
    #     a = 1
    #     raise ValueError(f"FUCK. Mismatched: {out.numel() - (out == my_out).sum().item()}")
    return out


def _my_dequantize_bf16(W, quant_state):
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

    # out_absmax = my_dequantize_blockwise_fp32_triton(uint8_lookup, absmax, absmax2, blocksize_uint8, n_absmax)
    # out_absmax += offset
    # #
    # out = my_cdequantize_blockwise_bf16_nf4_triton(W, out_absmax, blocksize_nf4, out.numel(), shape)
    my_out, absmax_dequantized = my_combined_dequantize(W, absmax, uint8_lookup, absmax2, blocksize_uint8,
                                                        blocksize_nf4, n_absmax,
                                                        out.numel(), offset, shape)

    return my_out


# ------------------------------- COMBINED KERNEL ------------------------------------------


@triton.jit
def combined_dequant_kernel(
        W_ptr,
        absmax_quantized_ptr,
        uint8_lookup_ptr,
        absmax2_ptr,
        out_ptr,
        absmax_dequantized_ptr,
        nf4_table_ptr,
        blocksize_uint8: tl.constexpr,  # number of scale values computed per block
        blocksize_nf4: tl.constexpr,  # not directly used below; kept for interface consistency
        n_absmax: tl.constexpr,  # total number of absmax quantized values
        n_out: tl.constexpr,  # total number of output elements (not used directly here)
        offset: tl.constexpr,  # constant offset (Python scalar)
        nf4_blocksize_to_process_per_block: tl.constexpr,  # number of bytes to process in NF4 stage per block
):
    pid = tl.program_id(0)
    block_start = pid * blocksize_uint8
    offsets = block_start + tl.arange(0, blocksize_uint8)
    mask = offsets < n_absmax

    a_vals = tl.load(absmax_quantized_ptr + offsets, mask=mask)
    indices = tl.cast(a_vals, tl.int32)

    scaling = tl.load(absmax2_ptr + pid)
    dequant_values = tl.load(uint8_lookup_ptr + indices)
    out_absmax_vec = dequant_values * scaling + offset

    # -------------------------------------------------
    # Stage 2: Dequantize the NF4–quantized weights.
    # -------------------------------------------------

    nf4_block_start = pid * nf4_blocksize_to_process_per_block
    out_block_start = pid * 2 * nf4_blocksize_to_process_per_block

    idx = tl.arange(0, nf4_blocksize_to_process_per_block)
    w_val = tl.load(W_ptr + nf4_block_start + idx).cast(tl.uint8)

    high = w_val >> 4
    low = w_val & 0x0F

    nf4_high = tl.load(nf4_table_ptr + high)  # Size=8192
    nf4_low = tl.load(nf4_table_ptr + low)

    nf4_high = nf4_high.reshape(blocksize_uint8, nf4_blocksize_to_process_per_block // blocksize_uint8).trans()
    nf4_low = nf4_low.reshape(blocksize_uint8, nf4_blocksize_to_process_per_block // blocksize_uint8).trans()

    result_high = nf4_high * out_absmax_vec
    result_low = nf4_low * out_absmax_vec

    result_high = result_high.trans().reshape(nf4_blocksize_to_process_per_block).cast(tl.float32)
    result_low = result_low.trans().reshape(nf4_blocksize_to_process_per_block).cast(tl.float32)

    out_idx = out_block_start + idx * 2
    tl.store(out_ptr + out_idx, tl.cast(result_high, tl.bfloat16))
    tl.store(out_ptr + out_idx + 1, tl.cast(result_low, tl.bfloat16))


def my_combined_dequantize(W, absmax_quantized, uint8_lookup, absmax2, blocksize_uint8, blocksize_nf4, n_absmax, n_out,
                           offset, shape):
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

    # I'm launching 1024 blocks.
    grid = (n_absmax // blocksize_uint8,)  # 1024. Original grid size for 2nd function was 256k
    # Each block in the grid has to process 256 blocks of W
    # Each block has to process 8192 elements of W
    nf4_table = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                              0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=torch.float32,
                             device=W.device)

    out = torch.empty(n_out, dtype=torch.bfloat16, device=W.device)
    absmax_dequantized = torch.empty(n_absmax, dtype=torch.float32, device=W.device)
    nf4_blocksize_to_process_per_block = (W.numel() // (n_absmax // blocksize_uint8))

    combined_dequant_kernel[grid](W, absmax_quantized, uint8_lookup, absmax2, out, absmax_dequantized,
                                  nf4_table,
                                  blocksize_uint8,
                                  blocksize_nf4, n_absmax, n_out, offset.item(), nf4_blocksize_to_process_per_block)

    out = out.view(shape)
    return out, absmax_dequantized


# ------------------------------- UNIT8 Dequantization -------------------------------------------


@triton.jit
def dequantize_kernel(A_ptr, code_ptr, absmax_ptr, out_ptr, blocksize: tl.constexpr, n: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * blocksize
    offsets = block_start + tl.arange(0, blocksize)
    mask = offsets < n

    a_vals = tl.load(A_ptr + offsets, mask=mask)
    indices = tl.cast(a_vals, tl.int32)

    scaling = tl.load(absmax_ptr + pid)
    dequant_values = tl.load(code_ptr + indices)
    result = dequant_values * scaling

    tl.store(out_ptr + offsets, result, mask=mask)


def my_dequantize_blockwise_fp32_triton(code, A, absmax, blocksize, n):
    out = torch.empty(n, dtype=torch.float32, device=A.device)
    grid = (n // blocksize,)
    dequantize_kernel[grid](A, code, absmax, out, blocksize, n)
    return out


# ------------------------------- NF4 Dequantization -----------------------

@triton.jit
def dequantize_kernel_nf4(A, absmax, out, nf4_table, blocksize: tl.constexpr):
    pid = tl.program_id(0)
    a_start = pid * (blocksize // 2)
    out_start = pid * blocksize

    idx = tl.arange(0, blocksize // 2)

    a_ptr = A + a_start + idx
    a_vals = tl.load(a_ptr)

    high = a_vals >> 4
    low = a_vals & 0x0F

    scale = tl.load(absmax + pid)

    deq_high = tl.load(nf4_table + high)
    deq_low = tl.load(nf4_table + low)

    result_high = deq_high * scale
    result_low = deq_low * scale

    # tl.device_print(result_high.dtype)
    out_idx = out_start + idx * 2
    tl.store(out + out_idx, tl.cast(result_high, tl.bfloat16))
    tl.store(out + out_idx + 1, tl.cast(result_low, tl.bfloat16))


def my_cdequantize_blockwise_bf16_nf4_triton(A, absmax, blocksize, n, shape):
    nf4_table = torch.tensor(
        [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
         -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
         0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
         0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0],
        device=A.device, dtype=torch.float32
    )

    out = torch.empty(n, dtype=torch.bfloat16, device=A.device)

    grid = (A.numel() // (blocksize // 2),)

    dequantize_kernel_nf4[grid](A, absmax, out, nf4_table, blocksize)

    return out.view(shape)
