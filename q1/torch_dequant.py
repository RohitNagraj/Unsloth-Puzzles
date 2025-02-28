import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nf4_data = torch.tensor(
    [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
     -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
     0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], device=device)


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

    out_absmax = my_dequantize_blockwise_fp32(
        code2, absmax, absmax2, blocksize2, n_elements_absmax)

    out_absmax += offset

    out = cdequantize_blockwise_bf16_nf4(
        W, out_absmax, blocksize, out.numel(), out.shape)

    return out


def my_dequantize_blockwise_fp32(code, A, absmax, blocksize, n):
    """
    Dequantizes unit8 quantized array to torch.float32.
    :param code: Array of type torch.float32 used as a lookup table.
    :param A: The array to dequantize. Currently quantized in uint8.
    :param absmax: The absolute maximum values used to scale the dequantized array.
    :param blocksize: The size of a chunk that uses the same absmax value for scaling.
    :param n: Number of elements in the output array.
    :return: Dequantized torch.float32 array.
    """
    my_out_absmax = torch.empty(
        n, dtype=torch.float32, device="cuda:0", requires_grad=False)

    for i in range(n // blocksize):
        for block in range(blocksize):
            my_out_absmax[i * blocksize +
                          block] = code[int(A[i * blocksize + block])] * absmax[i]
    return my_out_absmax


def cdequantize_blockwise_bf16_nf4(A, absmax, blocksize, n, shape):
    my_out = torch.empty(n, dtype=torch.bfloat16,
                         device="cuda:0", requires_grad=False)

    for block in range(A.numel() // (blocksize // 2)):
        for j in range(blocksize // 2):

            my_out[2 * ((block * (blocksize//2)) + j)] = nf4_data[(
                int(A[(block * (blocksize//2)) + j]) >> 4)] * absmax[block]

            my_out[2 * ((block * (blocksize//2)) + j) + 1] = nf4_data[int(
                A[(block * (blocksize//2)) + j]) & 0x0F] * absmax[block]

    return my_out.reshape(shape)

# The following is the CUDA kernels from bitsandbytes who's behaviour I'm replicating here. Keeping the reference implementation around to lookup functionality and optimizations.


"""
void cdequantize_blockwise_bf16_nf4(float *code(None), unsigned char *A (W), float *absmax (out_absmax), __nv_bfloat16 *out, int blocksize (64), 
  const int n(16777216), cudaStream_t stream){ dequantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n, stream); }
"""

"""
void dequantizeBlockwise_bf16_nf4(float *code (None), unsigned char *A (W), float *absmax(out_absmax), __nv_bfloat16 *out, int blocksize(64), 
  const int n(16777216), cudaStream_t stream){ dequantizeBlockwise<__nv_bfloat16, NF4>(NULL, A, absmax, out, blocksize, n, stream); }
"""

"""
template<typename T(nv_bf16), int DATA_TYPE(NF4, 2)> void dequantizeBlockwise(float *code (NULL), unsigned char *A (W), float *absmax, T *out, int blocksize(64), const int n(16777216), cudaStream_t stream)
{
  // printf("stream==%d\n",stream);
  int num_blocks = n/blocksize; = 262144
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512; = 1024
  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T(nv_bf16), 512, 64, 8, DATA_TYPE(2)><<<(n+tile_size-1)/tile_size(16384), 64, 0, stream>>>(code, A, absmax, out, blocksize/2(32), n(16777216));
  else
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64, 0, stream>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}
"""

"""
template<typename T(nv_bf16), int TILE_SIZE(512), int THREADS(64), int NUM_PER_TH(8), int DATA_TYPE(2)>
__global__ void kDequantizeBlockwise(float *code(None), unsigned char * A(W), float * absmax(out_absmax), T *out, const int blocksize(32), const int n(16777216))
{

  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)]; -> bf16 vals[16]
  unsigned char qvals[NUM_PER_TH]; qvals[8]
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE)
  {
    if (DATA_TYPE > 0)
    {
      valid_items_load = min(TILE_SIZE, (n + 1) / 2 - i);
      valid_items_store = min(TILE_SIZE * 2, n - i * 2);
    }
    else
    {
      valid_items_load = min(TILE_SIZE, n - i);
      valid_items_store = valid_items_load;
    }

    // Since blocksize will always be a power-of-2, we avoid more expensive
    // division by the blocksize and instead use a shift operation.
    // This is equivalent to (i+threadId.x*NUM_PER_TH)/blocksize.
    local_abs_max = __ldg(&absmax[(i+threadIdx.x*NUM_PER_TH) >> (31 - __clz(blocksize))]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

    switch (DATA_TYPE)
    {
        case General8bit:
          // load code through read-only cache via __ldg
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
            vals[j] = __ldg(&code[qvals[j]])*local_abs_max;
          break;
        case FP4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
            vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
          }
          break;
        case NF4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
            vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max;
            dDequantizeNF4 just does a lookup of the index at the inner val.
          }
          break;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
  }
}
"""
