import os
import shutil
import subprocess
import sys

KERNELS = [
    {
        "name": "Vector Addition",
        "source": "cuda-add.cu",
        "binary": "cuda-add",
        "chapter": "Chapters 2 & 3 (PMPP)"
    },
    {
        "name": "Basic Matrix Multiplication",
        "source": "cuda-matrix-mult.cu",
        "binary": "cuda-matrix-mult",
        "chapter": "Chapter 3 (PMPP)"
    },
    {
        "name": "Tiled Matrix Multiplication (Shared Memory)",
        "source": "cuda-matrix-mult-tile.cu",
        "binary": "cuda-matrix-mult-tile",
        "chapter": "Chapters 4 & 5 (PMPP)"
    },
]


def find_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]
    for p in common_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def run_command(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout, res.stderr


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("  CUDA Workbook Kernel Runner (PMPP)")
    print("=" * 60)

    nvcc = find_nvcc()
    if not nvcc:
        print("\n[WARNING] 'nvcc' (NVIDIA CUDA Compiler) was not found on PATH.")
        print("To compile and run CUDA C++ kernels, please run this on a system with an NVIDIA GPU and CUDA Toolkit installed (e.g., Google Colab, Linux GPU workstation, RunPod).\n")
        print("Available kernel files in this repository:")
        for k in KERNELS:
            print(f"  • {k['name']}: {k['source']} ({k['chapter']})")
        return

    print(f"Using NVCC compiler: {nvcc}\n")

    for k in KERNELS:
        src = os.path.join(root_dir, k["source"])
        bin_path = os.path.join(root_dir, k["binary"])

        if not os.path.exists(src):
            print(f"Skipping {k['name']}: source file {k['source']} not found.")
            continue

        print("-" * 60)
        print(f"Compiling: {k['name']} [{k['source']}]")
        compile_cmd = f"{nvcc} -O3 -o {bin_path} {src}"
        code, out, err = run_command(compile_cmd, cwd=root_dir)

        if code != 0:
            print(f"Compilation FAILED for {k['source']}:")
            print(err)
            continue
        print("Compilation successful! Running executable:")

        run_cmd = f"./{k['binary']}"
        code, out, err = run_command(run_cmd, cwd=root_dir)
        if code != 0:
            print(f"Execution failed with code {code}:")
            print(err)
        else:
            print(out.strip())

    print("\n" + "=" * 60)
    print("All kernel executions completed.")
    print("=" * 60)


if __name__ == '__main__':
    main()
