import sys
import time
import torch

try:
    from torch.utils.cpp_extension import load_inline
except ImportError:
    load_inline = None

cuda_source = """
#include <torch/extension.h>

#define TILE_WIDTH 16

__global__
void matrixMult(int *A, int *B, int *C, int inner_dim, int w, int h)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < w && row < h)
    {
        int sum = 0;
        for (int i = 0; i < inner_dim; i++) {
            sum += A[row * w + i] * B[i * w + col];
        }
        C[row * w + col] = sum;
    }
}

void run_matrixmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int width) {
    int* A_ptr = A.data_ptr<int>();
    int* B_ptr = B.data_ptr<int>();
    int* C_ptr = C.data_ptr<int>();

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    int blockX = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int blockY = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blocks(blockX, blockY);

    matrixMult<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, width, width, width);
}

__global__ 
void matmul(int *A, int *B, int *C, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int m_ds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int n_ds[TILE_WIDTH][TILE_WIDTH];

    int sum = 0;
    for (int i = 0; i * TILE_WIDTH < width; i++)
    {
        m_ds[threadIdx.y][threadIdx.x] = A[row * width + (i * TILE_WIDTH + threadIdx.x)];
        n_ds[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * width + col];
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            sum += m_ds[threadIdx.y][j] * n_ds[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * width + col] = sum;
}

void run_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int width) {
    int* A_ptr = A.data_ptr<int>();
    int* B_ptr = B.data_ptr<int>();
    int* C_ptr = C.data_ptr<int>();

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    int blockX = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int blockY = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blocks(blockX, blockY);

    matmul<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, width);
}
"""

cpp_source = """
void run_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int width);
void run_matrixmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int width);
"""


def benchmark(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. A CUDA-capable GPU is required to compile and run these kernels.")
        return

    print("Compiling inline CUDA kernels with PyTorch load_inline...")
    cuda_module = load_inline(
        name='custom_matmul_v3',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['run_matmul', 'run_matrixmul'],
        verbose=True
    )
    print("Compilation successful!\n")

    N = 1024
    print(f"Matrix Dimension: {N}x{N}")

    A_tensor = torch.randint(0, 10, (N, N), dtype=torch.int32, device='cuda')
    B_tensor = torch.randint(0, 10, (N, N), dtype=torch.int32, device='cuda')
    C_no_tile = torch.zeros((N, N), dtype=torch.int32, device='cuda')
    C_tiled = torch.zeros((N, N), dtype=torch.int32, device='cuda')

    A_float = A_tensor.float()
    B_float = B_tensor.float()

    print("\nRunning benchmarks...")
    pytorch_time = benchmark(lambda: torch.matmul(A_float, B_float))
    print(f"  PyTorch FP32 GEMM:            {pytorch_time:.3f} ms")

    no_tile_time = benchmark(lambda: cuda_module.run_matrixmul(A_tensor, B_tensor, C_no_tile, N))
    print(f"  Custom Kernel (No Tiling):    {no_tile_time:.3f} ms")

    tile_time = benchmark(lambda: cuda_module.run_matmul(A_tensor, B_tensor, C_tiled, N))
    print(f"  Custom Kernel (Shared Tile):  {tile_time:.3f} ms")

    # Correctness verification
    expected_C = torch.matmul(A_float, B_float).to(torch.int32)
    correct_no_tile = torch.equal(C_no_tile, expected_C)
    correct_tile = torch.equal(C_tiled, expected_C)

    print("\nCorrectness Verification:")
    print(f"  No Tiling accurate: {correct_no_tile}")
    print(f"  Shared Tile accurate: {correct_tile}")


if __name__ == '__main__':
    main()
