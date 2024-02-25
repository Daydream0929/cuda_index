/*
github: https://github.com/Daydream0929/cuda_index
blog: https://daydream0929.github.io/深入浅出cuda索引.html
*/

__global__ void test()
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bid = by * blockDim.x + bx;
    const int block_tid = tz * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;
    const int global_tid = bid * (blockDim.x * blockDim.y * blockDim.z) + block_tid;
    printf("bid: %d  --- block_tid : %d  --- globa_tid : %d\n", bid, block_tid, global_tid);
}

int main()
{
    dim3 blocks_per_grid = {2, 2, 1};
    dim3 threads_per_block = {2, 3, 4};
    test<<<blocks_per_grid, threads_per_block>>>();
    cudaDeviceSynchronize();
    return 0;
}
