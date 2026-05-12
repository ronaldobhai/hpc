# ass4_2_matrix_mul.cu - Line by Line Explanation

This CUDA program multiplies two square matrices on the GPU.

## 1. Build and run comments
- Line 1 shows the compile command using `nvcc`.
- Line 2 shows how to run the compiled program.

## 2. Library includes
- `#include <bits/stdc++.h>` brings in most standard C++ headers.
- `#include <iostream>` is used for input and output.
- `#include <cuda_runtime.h>` is used for CUDA memory functions and kernel support.

## 3. Namespace
- `using namespace std;` lets the code use standard names without `std::`.

## 4. CUDA kernel definition
- `__global__ void matrixMultiply(...)` defines a GPU kernel.
- `__global__` means the function is launched from the CPU and executed on the GPU.
- The kernel receives matrices A, B, C, and the matrix size N.

## 5. Thread position
- `int row = blockIdx.y * blockDim.y + threadIdx.y;` computes the row index.
- `int col = blockIdx.x * blockDim.x + threadIdx.x;` computes the column index.
- Together, row and col show which output cell this thread will compute.

## 6. Bounds check
- `if (row < N && col < N)` makes sure the thread is inside the matrix.
- This is needed because CUDA can launch extra threads.
- Without this check, threads could write outside the array.

## 7. Dot product calculation
- `int sum = 0;` starts the total for one output cell.
- The `for (int k = 0; k < N; k++)` loop walks across one row of A and one column of B.
- `sum += A[row * N + k] * B[k * N + col];` multiplies matching elements and adds them.
- This is the standard matrix multiplication rule.
- The final value is written into `C[row * N + col]`.

## 8. main function setup
- `int N = 4;` sets the matrix size to 4x4.
- `int size = N * N * sizeof(int);` computes how many bytes are needed.
- The program works only with square matrices in this version.

## 9. Host memory allocation
- `int *A, *B, *C;` are CPU-side matrix pointers.
- `cudaMallocHost()` allocates pinned memory for faster host-device copies.
- A, B, and C are used for the two inputs and the result.

## 10. Matrix initialization
- The loop fills A and B with values `1, 2, 3, ...`.
- This gives the program a simple test case.
- The output can be checked manually if needed.

## 11. Device memory allocation
- `int *dev_A, *dev_B, *dev_C;` are GPU memory pointers.
- `cudaMalloc()` allocates space on the GPU.
- These arrays will be used by the kernel.

## 12. Copying data to the GPU
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` sends A and B to device memory.
- The GPU cannot use the host arrays directly in this version.

## 13. Launch configuration
- `dim3 threadsPerBlock(16, 16);` sets a 2D block of 16x16 threads.
- `dim3 numBlocks(...)` computes how many blocks are needed in x and y directions.
- The code prints the launch shape so the user can understand the GPU layout.
- 2D blocks fit matrix problems very well because matrices are 2D.

## 14. Kernel launch
- `matrixMultiply<<<numBlocks, threadsPerBlock>>>(...)` starts the GPU work.
- Each thread computes one output cell.
- This gives a lot of parallelism.

## 15. Copying the result back
- `cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);` moves the result from GPU to CPU memory.
- After this, C contains the final multiplied matrix.

## 16. Printing the matrix
- The nested loops print each row and column of C.
- This makes the output easy to read as a matrix.
- The result should show the full NxN product.

## 17. Freeing memory
- `cudaFree()` releases the GPU allocations.
- `cudaFreeHost()` releases the pinned host memory.
- This prevents memory leaks.

## 18. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This file does matrix multiplication on the GPU.
- One CUDA thread computes one cell of the result matrix.
- The code is a standard 2D CUDA example.