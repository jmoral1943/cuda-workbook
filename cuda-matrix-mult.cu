%%cuda
#include <iostream>
#include <cuda_runtime.h>


__global__ 
void matrixMult(int *A, int *B, int *C, int inner_dim, int w, int h)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // we need something to know the future m*n
    // we need to know the inner dimesion of the matrix a
    if (col < w && row < h)
    {
        int sum = 0;
        for (int i = 0; i < inner_dim; i++) {
            sum += A[row*w +i] * B[i*w + col];
        }

        C[row * w + col] = sum;
    }
}

// main function for running on the host
int main()
{
    // create the main host memory variables
    int A_h[100] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Row 0
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Row 1
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // Row 2
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // Row 3
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5  // Row 4
    };

    int B_h[100] = {
        1, 2, 3, 4, 5, // Row 0
        1, 2, 3, 4, 5, // Row 1
        1, 2, 3, 4, 5, // Row 2
        1, 2, 3, 4, 5, // Row 3
        1, 2, 3, 4, 5, // Row 4
        1, 2, 3, 4, 5, // Row 5
        1, 2, 3, 4, 5, // Row 6
        1, 2, 3, 4, 5, // Row 7
        1, 2, 3, 4, 5, // Row 8
        1, 2, 3, 4, 5, // Row 9
        1, 2, 3, 4, 5, // Row 10
        1, 2, 3, 4, 5, // Row 11
        1, 2, 3, 4, 5, // Row 12
        1, 2, 3, 4, 5, // Row 13
        1, 2, 3, 4, 5, // Row 14
        1, 2, 3, 4, 5, // Row 15
        1, 2, 3, 4, 5, // Row 16
        1, 2, 3, 4, 5, // Row 17
        1, 2, 3, 4, 5, // Row 18
        1, 2, 3, 4, 5  // Row 19
    };
    int C_h[100];

    // create the main device memory variables
    int *A_d, *B_d, *C_d;

    size_t array_size = sizeof(A_h);

    // allocate the device memory vairable
    cudaMalloc((void **)&A_d, array_size);
    cudaMalloc((void **)&B_d, array_size);
    cudaMalloc((void **)&C_d, array_size);
    // copy the memory from the host variabels to the device variables

    cudaMemcpy(A_d, A_h, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, array_size, cudaMemcpyHostToDevice);

    // call the kernel
    dim3 threads(5,5);
    dim3 blocks(1,1);

    matrixMult<<<blocks, threads>>>(A_d, B_d, C_d, 20, 5, 5);

    // transfer the memory from device to host
    cudaMemcpy(C_h, C_d, array_size, cudaMemcpyDeviceToHost);

    // print the new host variable

    for (int i = 0; i < 10; i++)
    {
        std::cout << C_h[i] << std::endl;
    }

    // clear the memory
    cudaFree(&A_d);
    cudaFree(&B_d);
    cudaFree(&C_d);

    return 0;
}