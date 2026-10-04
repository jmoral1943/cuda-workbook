#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

        // ==========================================
        // CHALLENGE 1: VECTOR ADDITION (PHASE 2A)
        // ==========================================

        // TODO: Write your __global__ vectorAdd kernel
        // 1. Calculate thread global index 'i' using blockIdx.x, blockDim.x, and threadIdx.x
        // 2. Add bounds check (i < numElements)
        // 3. Perform addition: C[i] = A[i] + B[i]

        __global__ void vectorAdd(int *A, int *B, int *C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    // TODO 1: Define host data (two 1D arrays A and B with at least 10 elements)
    int A_h[100] = {
        84, 12, 95, 37, 48, 61, 23, 79, 10, 52,
        91, 14, 63, 88, 29, 45, 76, 33, 58, 97,
        19, 82, 41, 67, 25, 93, 11, 54, 70, 36,
        89, 22, 47, 65, 31, 80, 15, 59, 94, 28,
        73, 17, 62, 38, 85, 21, 50, 96, 42, 69,
        13, 78, 24, 87, 35, 60, 18, 92, 49, 71,
        30, 83, 16, 55, 98, 27, 64, 40, 77, 20,
        90, 34, 57, 81, 26, 68, 43, 99, 15, 72,
        32, 86, 51, 66, 39, 95, 12, 74, 23, 56,
        88, 44, 70, 19, 82, 31, 63, 46, 91, 25};
    int B_h[100] = {
        16, 88, 5, 63, 52, 39, 77, 21, 90, 48,
        9, 86, 37, 12, 71, 55, 24, 67, 42, 3,
        81, 18, 59, 33, 75, 7, 89, 46, 30, 64,
        11, 78, 53, 35, 69, 20, 85, 41, 6, 72,
        27, 83, 38, 62, 15, 79, 50, 4, 58, 31,
        87, 22, 76, 13, 65, 40, 82, 8, 51, 29,
        70, 17, 84, 45, 2, 73, 36, 60, 23, 80,
        10, 66, 43, 19, 74, 32, 57, 1, 85, 28,
        68, 14, 49, 34, 61, 5, 88, 26, 77, 44,
        12, 56, 30, 81, 18, 69, 37, 54, 9, 75};
    int C_h[100] = {};
    // TODO 2: Calculate size in bytes
    size_t array_size = sizeof(A_h);

    // TODO 3: Declare device pointers (d_A, d_B, d_C) and allocate GPU memory using cudaMalloc
    int *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, array_size);
    cudaMalloc((void **)&B_d, array_size);
    cudaMalloc((void **)&C_d, array_size);

    // TODO 4: Copy host arrays to device using cudaMemcpy (cudaMemcpyHostToDevice)
    cudaMemcpy(A_d, A_h, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, array_size, cudaMemcpyHostToDevice);

    // TODO 5: Define block/grid dimensions and launch your kernel: kernelName<<<blocks, threads>>>(...)
    int threads = 256;
    int blocks = ceil(100 / 256.0);

    vectorAdd<<<blocks, threads>>>(A_d, B_d, C_d, 100);

    // TODO 6: Call cudaDeviceSynchronize() and copy result C back to CPU (cudaMemcpyDeviceToHost)
    cudaDeviceSynchronize();
    cudaMemcpy(C_h, C_d, array_size, cudaMemcpyDeviceToHost);
    // TODO 7: Print host result to verify math correctness
    for (int i = 0; i < 10; i++)
    {
        printf("%d + %d = %d\n", A_h[i], B_h[i], C_h[i]);
    }

    // TODO 8: Free device memory using cudaFree
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}
