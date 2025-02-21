"""
weight.weight -> 2M
weight.quant_state.absmax -> 64k


"""
import torch

from bitsandbytes.functional import get_ptr
from bitsandbytes.functional import lib
import ctypes

ctypes_c_int = ctypes.c_int


def my_dequantize(weight):
    return _my_dequantize_bf16(weight.weight, weight.weight.quant_state)


def _my_dequantize_bf16(W, quant_state):
    CUDA_STREAM = torch.cuda.current_stream("cuda:0")

    absmax = quant_state.absmax
    shape = quant_state.shape
    dtype = quant_state.dtype
    blocksize = quant_state.blocksize
    offset = quant_state.offset
    state2 = quant_state.state2
    absmax2 = state2.absmax
    code2 = state2.code
    blocksize2 = state2.blocksize

    n_elements_absmax = absmax.numel()
    out = torch.empty(shape, dtype=dtype, device="cuda:0", requires_grad=False)
    out_absmax = torch.empty(n_elements_absmax, dtype=torch.float32, device="cuda:0", requires_grad=False)
    ptr_out_absmax = get_ptr(out_absmax)

    lib.cdequantize_blockwise_fp32(get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
                                   ctypes_c_int(blocksize2), ctypes_c_int(n_elements_absmax), CUDA_STREAM)
    out_absmax += offset

    lib.cdequantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
                                       ctypes_c_int(blocksize), ctypes_c_int(out.numel()), CUDA_STREAM)

    return out

