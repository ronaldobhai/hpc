// Build: nvcc ass4_2_matrix_mul.cu -o ass4_2_matrix_mul
// Run: ./ass4_2_matrix_mul

#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

/* ================= CUDA KERNEL ================= */

__global__ void matrixMultiply(int* A, int* B, int* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        int sum = 0;

        for(int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

/* ================= MAIN ================= */

int main()
{
    int N = 4;

    int size = N * N * sizeof(int);

    // Host matrices
    int *A, *B, *C;

    // Allocate pinned host memory
    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    // Initialize matrices
    for(int i = 0; i < N * N; i++)
    {
        A[i] = i + 1;
        B[i] = i + 1;
    }

    // Device matrices
    int *dev_A, *dev_B, *dev_C;

    // Allocate device memory
    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    // Copy matrices from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Define block size
    dim3 threadsPerBlock(16, 16);

    // Define grid size
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print Result Matrix
    cout << "Result Matrix:\n";

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << C[i * N + j] << " ";
        }

        cout << endl;
    }

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