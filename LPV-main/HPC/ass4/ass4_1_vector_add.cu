// Build: nvcc ass4_1_vector_add.cu -o ass4_1_vector_add
// Run: ./ass4_1_vector_add

#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

/* ================= CUDA KERNEL ================= */

__global__ void addVectors(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}

/* ================= MAIN ================= */

int main()
{
    int n = 1000000;

    int size = n * sizeof(int);

    // Host pointers
    int *A, *B, *C;

    // Allocate pinned host memory
    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    // Initialize vectors
    for(int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    // Device pointers
    int *dev_A, *dev_B, *dev_C;

    // Allocate device memory
    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    // Copy data from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int blockSize = 256;

    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    cout << "First 10 Elements of Result Vector:\n";

    for(int i = 0; i < 10; i++)
    {
        cout << C[i] << " ";
    }

    cout << endl;

    // Free device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    // Free host memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}