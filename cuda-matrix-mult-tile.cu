%%cuda
#include <iostream>
#include <cuda_runtime.h>

// A standard CPU matrix multiplication for verification
void matrixMultCPU(int *A, int *B, int *C, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            int sum = 0;
            for (int i = 0; i < width; ++i) {
                // Flattened 1D index math
                sum += A[row * width + i] * B[i * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

// Compares the CPU result against the GPU result
bool verifyResults(int *cpu_result, int *gpu_result, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            std::cout << "Mismatch found at index " << i 
                      << "! CPU: " << cpu_result[i] 
                      << " | GPU: " << gpu_result[i] << "\n";
            return false; // Fail immediately on the first error
        }
    }
    std::cout << "SUCCESS! GPU results perfectly match the CPU.\n";
    return true;
}

#define TILE_WIDTH 16

// device function for matrix multiplication
__global__ 
void matrixmult(int *A, int *B, int *C, int width, int height)
{
    // get y and x
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // create shared memory variable 2d array
    __shared__ int m_ds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int n_ds[TILE_WIDTH][TILE_WIDTH];

    int sum = 0;
    // make the loop for width of the array / title width
    for (int i = 0; i * TILE_WIDTH < width; i++)
    {
        // load m and n into shared memory
        m_ds[threadIdx.x][threadIdx.y] = 
         (i *TILE_WIDTH +threadIdx.x < width) ? A[row * width + (i *TILE_WIDTH +threadIdx.x)] : 0;
        n_ds[threadIdx.x][threadIdx.y]  = 
         (i *TILE_WIDTH +threadIdx.y < height) ? B[(i *TILE_WIDTH +threadIdx.y) * width + col] : 0;
        // sync threads
        __syncthreads();
        // from i to title width, load all the numbers that other threads have loaded into shared memory
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            // mutltiple m[i] n[i] then add to the sum
            sum += m_ds[j][threadIdx.y] * n_ds[threadIdx.x][j];
        }
        // sync threads
        __syncthreads();
    }
    // load the sum at C[x][y]
    C[row *width + col] = sum;
}

// host function
int main()
{
    int N = 1024; // Change this to scale up your test!
    size_t bytes = N * N * sizeof(int);

    // 1. Allocate Heap Memory (Do NOT use standard arrays like int A_h[1024]; the stack will crash!)
    int *A_h = (int *)malloc(bytes);
    int *B_h = (int *)malloc(bytes);
    int *C_h = (int *)malloc(bytes);

    // 2. Instantly fill the matrices with random numbers between 0 and 9
    for (int i = 0; i < N * N; i++)
    {
        A_h[i] = rand() % 10;
        B_h[i] = rand() % 10;
    }

    // allocate the memory on device
    int *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, bytes);
    cudaMalloc((void **)&B_d, bytes);
    cudaMalloc((void **)&C_d, bytes);

    // transfer the memory from host to device
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

  
    // calculate the tiles
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

    int blocksX = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocksY = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 blocksPerGrid(blocksX, blocksY,1);

    // call the kernel
    matrixmult<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N, N);
                                
    // transfer the data to the local variable `C`
    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    // print the data
                                
    // 1. Allocate a new array for the CPU to do its own math
    int *C_cpu = (int*)malloc(bytes);

    // 2. Run the CPU version
    std::cout << "Calculating CPU reference...\n";
    matrixMultCPU(A_h, B_h, C_cpu, N);

    // 3. Verify they match
    std::cout << "Verifying results...\n";
    verifyResults(C_cpu, C_h, N * N);

    // 4. Free the memory
    free(C_cpu);
    // deallocate the memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // return 0
    return 0;
}