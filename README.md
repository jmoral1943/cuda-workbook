# 🚀 CUDA Workbook (PMPP)

A hands-on CUDA C++ and PyTorch extension workbook based on ***Programming Massively Parallel Processors: A Hands-on Approach* (PMPP)** by David B. Kirk and Wen-mei W. Hwu.

This repository implements foundational parallel algorithms from the textbook, transitioning from simple 1D vector addition to 2D matrix multiplication and high-throughput tiled matrix multiplication utilizing GPU shared memory, along with PyTorch inline C++/CUDA kernel benchmarking.

---

## 📚 Curriculum & Implemented Kernels

| File | PMPP Chapters | Topic / Concept | Key Mechanics |
| :--- | :--- | :--- | :--- |
| **`cuda-add.cu`** | Chapters 2 & 3 | **Vector Addition** | 1D thread indexing (`blockDim.x * blockIdx.x + threadIdx.x`), boundary checks, host-to-device memory transfers (`cudaMemcpy`). |
| **`cuda-matrix-mult.cu`** | Chapter 3 | **Basic Matrix Multiplication** | 2D thread/block indexing, global memory dot products, CPU verification baseline. |
| **`cuda-matrix-mult-tile.cu`** | Chapters 4 & 5 | **Tiled Matrix Multiplication** | Shared memory (`__shared__`), thread block barrier synchronization (`__syncthreads()`), memory coalescing, high arithmetic intensity. |
| **`cuda_performance.py`** | Advanced / PyTorch | **Performance Benchmarking** | Compiles custom CUDA kernels via PyTorch `load_inline` and benchmarks against cuBLAS / `torch.matmul`. |
| **`run_kernels.py`** | Tooling | **Automated Kernel Runner** | Automatically compiles and runs all standalone CUDA kernels with `nvcc`. |

---

## 🏗️ Architectural Concepts

### Tiled Matrix Multiplication (Shared Memory)
In naive global-memory matrix multiplication, each element is fetched repeatedly from high-latency DRAM. With **tiled matrix multiplication**, threads cooperatively load a tile of size $\text{TILE\_WIDTH} \times \text{TILE\_WIDTH}$ into fast on-chip shared memory:

$$\text{Arithmetic Intensity} = \mathcal{O}(\text{TILE\_WIDTH})$$

By reusing data across threads in the block and synchronizing with `__syncthreads()`, global memory traffic is drastically reduced by a factor of $\text{TILE\_WIDTH}$ (default `16x16`).

---

## 📓 Minimalist Notebooks

All Jupyter notebooks are configured as lightweight runners that execute the standalone Python and CUDA scripts:

*   **`CUDA_notebook.ipynb`**: Checks GPU hardware with `!nvidia-smi` and runs `run_kernels.py` to compile and verify all CUDA source files.
*   **`CUDA_performance.ipynb`**: Installs Ninja build system and runs `cuda_performance.py` to benchmark custom CUDA kernels against PyTorch FP32 matrix multiplication.

---

## 🛠️ Quickstart & Execution

### 1. Prerequisites
*   NVIDIA GPU with CUDA Toolkit installed (`nvcc`)
*   Python 3.8+ with PyTorch (for performance benchmarking)
*   `ninja` (recommended for fast inline C++/CUDA extension compilation)

```bash
pip install torch ninja
```

### 2. Run All Kernels via Python Runner

```bash
python run_kernels.py
```

### 3. Compile and Run Individual Kernels

```bash
# Vector Addition
nvcc -O3 -o cuda-add cuda-add.cu && ./cuda-add

# Basic Matrix Multiplication
nvcc -O3 -o cuda-matrix-mult cuda-matrix-mult.cu && ./cuda-matrix-mult

# Tiled Matrix Multiplication (Shared Memory)
nvcc -O3 -o cuda-matrix-mult-tile cuda-matrix-mult-tile.cu && ./cuda-matrix-mult-tile
```

### 4. Run PyTorch Performance Benchmark

```bash
python cuda_performance.py
```

---

## 📖 References
*   Kirk, D. B., & Hwu, W. W. *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann.
*   NVIDIA CUDA C++ Programming Guide.
