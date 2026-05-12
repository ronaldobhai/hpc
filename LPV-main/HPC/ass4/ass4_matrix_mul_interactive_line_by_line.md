# ass4_matrix_mul_interactive.cu - Line by Line Explanation

This is the user-input version of the CUDA matrix multiplication program.

## 1. Build and run comments
- Line 1 shows the compile command with `nvcc`.
- Line 2 shows how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard C++ headers.
- `#include <iostream>` is used for input and output.
- `#include <cuda_runtime.h>` is used for CUDA memory allocation and kernel launches.

## 3. Example comments
- The example input shows how to enter a 3x3 matrix test.
- The expected output shows what happens when the second matrix is an identity matrix.
- These comments are useful for quick testing.

## 4. Namespace
- `using namespace std;` keeps the code simple and readable.

## 5. CUDA kernel
- `__global__ void matrixMultiply(...)` defines the GPU function.
- The CPU launches this function, but the GPU runs it.
- It receives the two input matrices, the output matrix, and the size `N`.

## 6. Row and column calculation
- `int row = blockIdx.y * blockDim.y + threadIdx.y;` finds the output row for the thread.
- `int col = blockIdx.x * blockDim.x + threadIdx.x;` finds the output column.
- Together they identify one output cell.

## 7. Bounds check
- `if (row < N && col < N)` makes sure the thread is inside the matrix.
- This prevents memory errors when the GPU launches extra threads.

## 8. Matrix multiplication logic
- `int sum = 0;` starts the running total for one output cell.
- The inner loop walks through the row of A and the column of B.
- Each step multiplies one pair of values and adds it to sum.
- The final result is stored in C at the correct row and column.

## 9. Reading matrix size
- `int N;` stores the user-entered matrix size.
- The program asks for a square matrix size.
- If `N <= 0`, the program prints an error and exits.
- This avoids invalid memory allocation.

## 10. Memory size calculation
- `int size = N * N * sizeof(int);` gets the byte count for one matrix.
- CUDA memory calls work in bytes, not in element counts.

## 11. Allocating pinned host memory
- `int *A, *B, *C;` are CPU-side matrix pointers.
- `cudaMallocHost()` gives pinned memory, which is faster for copies.
- A and B store the input matrices.
- C stores the final product.

## 12. Reading matrix A
- The program asks the user to enter all `N*N` values for matrix A.
- The loop reads values row by row.
- The user enters the matrix in row-wise order.

## 13. Reading matrix B
- The program asks for matrix B next.
- Another loop reads all `N*N` values.
- This keeps the input format simple.

## 14. Allocating GPU memory
- `int *dev_A, *dev_B, *dev_C;` are device pointers.
- `cudaMalloc()` creates GPU memory for the matrices.
- The kernel will use these arrays.

## 15. Copying input to the GPU
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` sends A and B to device memory.
- The GPU needs those values before the kernel launch.

## 16. Launch configuration
- `dim3 threadsPerBlock(16, 16);` sets a 2D block of threads.
- `dim3 numBlocks(...)` calculates how many blocks are needed in x and y.
- The program prints the block grid and thread grid for learning.
- Matrix multiplication is naturally 2D, so this layout makes sense.

## 17. Kernel launch
- `matrixMultiply<<<numBlocks, threadsPerBlock>>>(...)` runs the GPU code.
- Each thread computes one cell of the output matrix.
- This is highly parallel.

## 18. Copying the result back
- `cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);` moves the product matrix back to the CPU.
- Now the host can print the answer.

## 19. Printing the result matrix
- The nested loops print matrix C row by row.
- This makes the output easy to read.
- For the identity-matrix example, the output should match matrix A.

## 20. Freeing memory
- `cudaFree()` releases GPU memory.
- `cudaFreeHost()` releases pinned host memory.
- This avoids memory leaks.

## 21. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This is the interactive CUDA matrix multiplication demo.
- The user enters both matrices.
- The GPU multiplies them and prints the full result.