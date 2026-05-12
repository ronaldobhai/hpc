# ass4_vector_add_interactive.cu - Line by Line Explanation

This is the user-input version of the CUDA vector addition program.

## 1. Build and run comments
- Line 1 tells you how to compile with `nvcc`.
- Line 2 tells you how to run the file.

## 2. Library includes
- `#include <bits/stdc++.h>` brings in most standard C++ features.
- `#include <iostream>` is for input and output.
- `#include <cuda_runtime.h>` is needed for CUDA memory and kernel calls.

## 3. Example comments
- The example input shows how to test the program.
- The expected output shows what the first few sums should look like.
- This makes the file easier to learn from.

## 4. Namespace
- `using namespace std;` lets the code write `cout` and `cin` directly.

## 5. CUDA kernel
- `__global__ void addVectors(...)` defines the GPU function.
- The CPU launches it, but the GPU executes it.
- The kernel gets the two input vectors, the output vector, and the size.

## 6. Thread index calculation
- `int i = blockIdx.x * blockDim.x + threadIdx.x;` gives each thread a unique element index.
- This is the normal CUDA way to map threads to data.

## 7. Bounds check
- `if (i < n)` prevents threads from going past the end of the array.
- CUDA often launches extra threads, so this check is necessary.

## 8. Actual vector addition
- `C[i] = A[i] + B[i];` adds one element from each vector.
- One thread handles one addition.
- This is a perfect fit for GPU parallelism.

## 9. Reading the vector size
- `int n;` stores the user-entered size.
- The program asks for the size.
- If the size is less than or equal to zero, the program stops with an error.
- This avoids bad memory allocation.

## 10. Calculating memory size
- `int size = n * sizeof(int);` turns the element count into bytes.
- CUDA memory calls use byte counts.

## 11. Allocating pinned host memory
- `int *A, *B, *C;` are CPU-side pointers.
- `cudaMallocHost()` allocates pinned memory for them.
- Pinned memory speeds up transfers between CPU and GPU.

## 12. Reading vector A
- The program asks the user to enter vector A.
- A loop reads one number at a time.
- Each value is stored directly in `A[i]`.

## 13. Reading vector B
- The program asks the user to enter vector B.
- Another loop reads each number.
- The values are stored in `B[i]`.

## 14. Allocating device memory
- `int *dev_A, *dev_B, *dev_C;` are GPU pointers.
- `cudaMalloc()` reserves memory on the device.
- The kernel will use these arrays.

## 15. Copying input to the GPU
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` sends A and B to the GPU.
- The GPU cannot use host memory directly in this program.
- Copying is required before the kernel launch.

## 16. Choosing the launch configuration
- `int blockSize = 256;` sets the number of threads per block.
- `int numBlocks = (n + blockSize - 1) / blockSize;` calculates how many blocks are needed.
- The program prints the chosen blocks and threads.
- This helps the user understand the GPU setup.

## 17. Launching the kernel
- `addVectors<<<numBlocks, blockSize>>>(...)` starts the GPU work.
- Each thread computes one sum.
- The GPU executes many threads at the same time.

## 18. Copying the result back
- `cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);` brings the result back to the CPU.
- Now the host array C holds the final summed values.

## 19. Printing the result vector
- The program prints all values in C.
- For correct input, each value should equal the matching A and B sum.
- For the example input, the output should be `0 3 6 9 12 15 18 21 24 27`.

## 20. Freeing memory
- `cudaFree()` releases GPU memory.
- `cudaFreeHost()` releases pinned host memory.
- These calls prevent leaks.

## 21. Final return
- `return 0;` finishes the program.

## Simple summary
- This file is the interactive GPU vector addition demo.
- The user provides both vectors.
- The GPU adds them one element at a time.