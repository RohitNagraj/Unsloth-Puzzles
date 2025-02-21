
# def cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, n):
#     """
#
#     :param code: torch float tensor -> torch array, 256 len, fp32
#     :param A: unsigned char array -> Quantized Data -> 262144 len, uint8
#     :param absmax: float array -> Absmax -> 1024 len -> fp32 -> Every block of 256 values has one absmax
#     :param out: float array -> output of dequantized absmax
#     :param blocksize: int: 256
#     :param n: const int n: 262144
#     :param CUDA_STREAM:
#     """
#
#     num_blocks = int(n / blocksize)
#     # num_blocks = n % blocksize == 0 ? num_blocks: num_blocks + 1;
#     if n % blocksize == 0:
#         num_blocks = num_blocks
#     else:
#         num_blocks = num_blocks + 1
#     # General8Bit = 0
#     tile_size = 512
    # kernel(code, A, absmax, out, blocksize, n) # Grid, block = (n+tile_size-1)/tile_size, 64)
    # Each block is processing partial absmax block.


#     Each grid handles 1 tile of 512 values.
#     Each block handles 8 values

# def kernel(code, A, absmax, out, blocksize, n):
#     n_load = gridDim.x * 512
#     lets assume gridDim.x = 1
#     n_load = 512
#     valid_items_load = 0
#     valid_items_store = 0
#     base_idx = blockIdx.x * 512
#     lets assume blockIdx.x = 0
#     base_idx = 0
#
#     float vals[8];
#     unsigned char qvals[8]
#     local_abs_max = -inf;
#
#     for (i = base_idx, i<n_load, i+= gridDim.x*512):
#         valid_items_load = min(512, n-i)
#         valid_items_store= valid_items_load
#         local_abs_max = absmax[(i+threadId.x*NUM_PER_TH)/blocksize]
#     syncthreads;
#     LoadChar(loadchar).Load( & (A[i]), qvals, valid_items_load, 128);
#
#     for (int j = 0; j < NUM_PER_TH; j++)
#         vals[j] = __ldg( & code[qvals[j]])*local_abs_max;
#       dequant[j] = code[quant[j]] * local_absmax; # Very very important.
#
#     syncthreads;
#     StoreT(storet).Store( & (out[(DATA_TYPE > 0) ? i * 2: i]), vals, valid_items_store);


"""
  void cdequantize_blockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n,
   cudaStream_t stream){ dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n, stream); }
"""

"""
void dequantizeBlockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, 
    cudaStream_t stream){ dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n, stream); }
"""

"""
template<typename T(float), int DATA_TYPE(General8bit)> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T(float) *out, int blocksize, const int n, cudaStream_t stream)
{
  // printf("stream==%d\n",stream);
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64, 0, stream>>>(code, A, absmax, out, blocksize/2, n);
  else
    kDequantizeBlockwise<T (float), 512, 64, 8, DATA_TYPE(General8Bit/0)><<<(n+tile_size-1)/tile_size (Grid), 64 (block), 0, stream>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

"""

"""
template<typename T(float), int TILE_SIZE(512), int THREADS(64), int NUM_PER_TH(8), int DATA_TYPE(0)>
__global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n)
{

  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
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
          }
          break;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
  }
}
"""



# def my_dequantize_blockwise_fp32(code, A, absmax, blocksize, n):
#     """
#     Dequantizes unit8 quantized array to torch.float32.
#     :param code: Array of type torch.float32 used as a lookup table.
#     :param A: The array to dequantize. Currently quantized in uint8.
#     :param absmax: The absolute maximum values used to scale the dequantized array.
#     :param blocksize: The size of a chunk that uses the same absmax value for scaling.
#     :param n: Number of elements in the output array.
#     :return: Dequantized torch.float32 array.
#     """
#     my_out_absmax = torch.empty(n, dtype=torch.float32, device="cuda:0", requires_grad=False)
#
#     for i in range(n // blocksize):
#         for block in range(blocksize):
#             my_out_absmax[i * blocksize + block] = code[int(A[i * blocksize + block])] * absmax[i]
#     return my_out_absmax


# def my_cdequantize_blockwise_bf16_nf4(A, absmax, blocksize, n, shape):
#     """
#     Dequantizes NF4 quantized array to torch.bfloat16
#     :param A: The NF4 array to dequantize
#     :param absmax: The absolute maximum values used to scale the dequantized array.
#     :param blocksize: The size of a chunk that uses the same absmax value for scaling.
#     :param n: Number of elements in the output array.
#     :param shape: The output shape of the dequantized array.
#     :return: Dequantized torch.Tensor of type torch.bfloat16 with the output shape specified.
#     """
#     nf4_data = torch.tensor(
#         [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
#          -0.18477343022823334,
#          -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
#          0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], device=device)
#     my_out = torch.empty(n, dtype=torch.bfloat16, device="cuda:0", requires_grad=False)
#
#     for block in range(A.numel() // (blocksize // 2)):
#         for j in range(blocksize // 2):
#             my_out[2 * ((block * (blocksize // 2)) + j)] = nf4_data[(int(A[(block * (blocksize // 2)) + j]) >> 4)] * \
#                                                            absmax[block]
#             my_out[2 * ((block * (blocksize // 2)) + j) + 1] = nf4_data[int(A[(block * (blocksize // 2)) + j]) & 0x0F] * \
#                                                                absmax[block]
#     return my_out.reshape(shape)