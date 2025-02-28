## Introduction
This question is about implementing a fused triton kernel to dequantize NF4 quantization.

The following are the steps involved in dequantization, performed in parallel:
1. Dequantize one block of absmax from uint8 to fp32 using the `weight.weight.quant_state.absmax` as the data vector and `weight.weight.quant_state.state2.absmax` as the absmax for it and the `weight.weight.quant_state.state2.code` as the lookup table. Block size is 256 (static).
2. Now dequantize one block of weight.weight from NF4 (packed as uint8) to fp32 using the previously dequantized absmax and NF4 lookup table (found in QLoRA paper and bitsandbytes codebase). 

Why is this faster than bitsandbytes and Unsloth?
1. Fused: This implementation uses one kernel whereas theirs use two, one for uint8 absmax dequant and one for NF4 weight dequant.
2. Cache eviction stategies: The bitsandbytes implementation does not use any sort of cache eviction stategies. This implementation prioritizes certain data in cache over others, reducing cache miss.
3. Tuned TILE_SIZE: Since this kernel is specifically being tuned for NVIDIA T4, picking a fixed TILE_SIZE that works best with T4 provides performance gains over the generic BLOCK_SIZE used by bitsandbytes.


## How to run
Just run `python benchmark.py`. It uses the implementation from `triton_dequant_fused.py` to benchmark it against `unsloth.fast_dequantize`.

## Extra
`q1_profiled.zip` contains `q1_profiled.ncu-rep` which is the Nsight Compute profiling output, used to profile and optimize the kernel.