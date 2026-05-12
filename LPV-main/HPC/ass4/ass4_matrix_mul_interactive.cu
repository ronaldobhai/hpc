// Build: nvcc ass4_matrix_mul_interactive.cu -o ass4_matrix_mul_interactive
// Run: ./ass4_matrix_mul_interactive

#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>

// Example Input:
// Enter matrix size (N for NxN matrix): 3
// Enter matrix A (9 values row-wise): 1 2 3 4 5 6 7 8 9
// Enter matrix B (9 values row-wise): 1 0 0 0 1 0 0 0 1

// Expected Output:
// Result Matrix:
// 1 2 3
// 4 5 6
// 7 8 9

using namespace std;

__global__ void matrixMultiply(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main() {
    int N;
    cout << "Enter matrix size (N for NxN matrix): ";
    cin >> N;

    if (N <= 0) {
        cout << "Invalid size!\n";
        return 1;
    }

    int size = N * N * sizeof(int);

    int *A, *B, *C;

    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    cout << "\nEnter matrix A (" << N * N << " values, row-wise):\n";
    for (int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    cout << "Enter matrix B (" << N * N << " values, row-wise):\n";
    for (int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    int *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cout << "\nLaunching kernel with " << numBlocks.x << "x" << numBlocks.y
         << " blocks and " << threadsPerBlock.x << "x" << threadsPerBlock.y
         << " threads per block...\n";

    matrixMultiply<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    cout << "\nResult Matrix (C = A * B):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
