# ass4_1_vector_add.cu - Line by Line Explanation

This CUDA program adds two large integer vectors on the GPU.

## 1. Build and run comments
- Line 1 tells you to compile with `nvcc`.
- Line 2 tells you how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` brings in most standard C++ headers.
- `#include <iostream>` is used for output.
- `#include <cuda_runtime.h>` gives access to CUDA memory and kernel support.

## 3. Namespace
- `using namespace std;` lets the code use standard names without `std::`.

## 4. CUDA kernel
- `__global__ void addVectors(...)` defines a function that runs on the GPU.
- `__global__` means the function is launched from the CPU but executed on the GPU.
- The kernel receives three arrays: A, B, and C.
- It also receives `n`, the number of elements.

## 5. Kernel index calculation
- `int i = blockIdx.x * blockDim.x + threadIdx.x;` computes the global thread index.
- `blockIdx.x` gives the block number.
- `blockDim.x` gives the number of threads per block.
- `threadIdx.x` gives the thread number inside the block.
- Together they identify one element of the vector.

## 6. Kernel boundary check
- `if (i < n)` makes sure the thread is working on a valid element.
- This is needed because the number of threads can be larger than the vector length.
- Without this check, the program could read or write outside the array.

## 7. Vector addition inside the kernel
- `C[i] = A[i] + B[i];` adds one element from A and B and stores the result in C.
- Each thread does one addition.
- That is why GPU code is very fast for this kind of task.

## 8. main function input
- `int n;` stores the vector size.
- The program asks the user for the size.
- `cin >> n;` reads the value.
- If the size is not valid, the program prints an error and exits.
- This keeps the program safe.

## 9. Memory size calculation
- `int size = n * sizeof(int);` computes the number of bytes needed.
- CUDA memory functions use bytes, not element counts.

## 10. Host memory allocation
- `int *A, *B, *C;` declares the CPU-side pointers.
- `cudaMallocHost()` allocates pinned host memory.
- Pinned memory is faster for copying between CPU and GPU.
- A, B, and C store the input vectors and the final output.

## 11. Reading vector input
- The program asks the user to enter vector A.
- A loop reads `n` values into `A`.
- Then the program asks for vector B.
- Another loop reads the `n` values into `B`.
- This makes the file interactive instead of hardcoded.

## 12. Device memory allocation
- `int *dev_A, *dev_B, *dev_C;` declares GPU memory pointers.
- `cudaMalloc()` allocates space on the GPU for each vector.
- This is where the kernel will read and write data.

## 13. Copying data to the GPU
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` copies data from CPU memory to GPU memory.
- First A is copied.
- Then B is copied.
- The GPU needs the input data before the kernel runs.

## 14. Launch configuration
- `int blockSize = 256;` chooses 256 threads per block.
- `int numBlocks = (n + blockSize - 1) / blockSize;` computes how many blocks are needed.
- This formula rounds up so all elements are covered.
- The program prints the launch configuration for learning.

## 15. Kernel launch
- `addVectors<<<numBlocks, blockSize>>>(...)` launches the GPU kernel.
- Each GPU thread handles one vector element.
- CUDA automatically spreads the work across hardware threads.

## 16. Copying result back
- `cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);` copies the result from GPU memory back to CPU memory.
- Now the host array C contains the final answers.

## 17. Printing result
- The program prints all elements of the result vector.
- For a correct run, `C[i]` should equal `A[i] + B[i]` for every position.
- The output is easy to verify by hand for small inputs.

## 18. Freeing GPU memory
- `cudaFree()` releases the device arrays.
- This avoids memory leaks on the GPU.

## 19. Freeing host memory
- `cudaFreeHost()` releases the pinned host memory.
- This is the correct way to free memory allocated by `cudaMallocHost()`.

## 20. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This program does one simple GPU job: add two vectors.
- The CPU prepares the data, the GPU does the math, and the CPU prints the result.