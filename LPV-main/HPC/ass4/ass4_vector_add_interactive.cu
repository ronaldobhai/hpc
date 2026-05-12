// Build: nvcc ass4_vector_add_interactive.cu -o ass4_vector_add_interactive
// Run: ./ass4_vector_add_interactive

#include <bits/stdc++.h>
#include <iostream>
#include <cuda_runtime.h>

// Example Input:
// Enter vector size: 10
// Vector A: 0 1 2 3 4 5 6 7 8 9
// Vector B: 0 2 4 6 8 10 12 14 16 18

// Expected Output:
// First 10 Elements: 0 3 6 9 12 15 18 21 24 27

using namespace std;

__global__ void addVectors(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n;
    cout << "Enter vector size: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid size!\n";
        return 1;
    }

    int size = n * sizeof(int);

    int *A, *B, *C;

    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    cout << "\nEnter vector A (space-separated):\n";
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }

    cout << "Enter vector B (space-separated):\n";
    for (int i = 0; i < n; i++) {
        cin >> B[i];
    }

    int *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cout << "\nLaunching kernel with " << numBlocks << " blocks and " << blockSize
         << " threads per block...\n";

    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    cout << "\nResult Vector (C = A + B):\n";
    for (int i = 0; i < n; i++) {
        cout << C[i] << " ";
    }
    cout << "\n";

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
