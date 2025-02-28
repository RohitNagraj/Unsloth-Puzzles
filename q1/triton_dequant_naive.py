import os

# Uncomment the following lines for debug.
# os.environ['TRITON_DEBUG'] = '1'
# os.environ['TRITON_INTERPRET'] = '1'

import torch

import triton
import triton.language as tl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    out = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

    out_absmax = my_dequantize_blockwise_fp32_uint8(
        uint8_lookup, absmax, absmax2, blocksize_uint8, n_absmax)

    out_absmax += offset

    out = my_dequantize_blockwise_bf16_nf4(
        W, out_absmax, blocksize_nf4, out.numel(), out.shape)

    return out


# ------------------------------- UNIT8 Dequantization -------------------------------------------


@triton.jit
def dequantize_kernel_uint8(A_ptr, code_ptr, absmax_ptr, out_ptr, blocksize: tl.constexpr, n: tl.constexpr):
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


def my_dequantize_blockwise_fp32_uint8(code, A, absmax, blocksize, n):
    out = torch.empty(n, dtype=torch.float32, device=A.device)
    grid = (n // blocksize,)
    dequantize_kernel_uint8[grid](A, code, absmax, out, blocksize, n)
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

    out_idx = out_start + idx * 2
    tl.store(out + out_idx, tl.cast(result_high, tl.bfloat16))
    tl.store(out + out_idx + 1, tl.cast(result_low, tl.bfloat16))


def my_dequantize_blockwise_bf16_nf4(A, absmax, blocksize, n, shape):
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
