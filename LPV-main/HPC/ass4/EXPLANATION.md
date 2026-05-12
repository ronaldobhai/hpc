# Assignment 4: CUDA Programming - Code Explanation

## Part 1: Vector Addition (ass4_1_vector_add.cu)

### Overview
This program adds two vectors using NVIDIA CUDA on the GPU, demonstrating fundamental GPU programming concepts.

---

## Header Files

```cpp
#include <iostream>         // Standard input/output
#include <cuda_runtime.h>   // CUDA runtime API
```

---

## CUDA Kernel: addVectors

### Kernel Definition
```cpp
__global__ void addVectors(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
    
    if(i < n)                                        // Boundary check
    {
        C[i] = A[i] + B[i];                         // Perform addition
    }
}
```

**Keywords:**
- **`__global__`**: Function runs on GPU, called from CPU

**Thread Indexing:**
```
Global thread index = (blockIdx.x × blockDim.x) + threadIdx.x
```

**Visualization:**
```
Grid:     [Block 0] [Block 1] [Block 2] ...
Block 0:  [T0][T1][T2]...[T255]
Block 1:  [T0][T1][T2]...[T255]

For Thread (blockIdx.x=1, threadIdx.x=5):
  Global index = (1 × 256) + 5 = 261
```

**Boundary check:**
```cpp
if(i < n)  // Prevent out-of-bounds access
```

---

## Main Function

### Step 1: Define Problem Size
```cpp
int n = 1000000;              // 1 million elements
int size = n * sizeof(int);   // Total memory size in bytes
```

---

### Step 2: Allocate Host Memory (Pinned)
```cpp
int *A, *B, *C;               // Host pointers

cudaMallocHost((void**)&A, size);  // Pinned memory for vector A
cudaMallocHost((void**)&B, size);  // Pinned memory for vector B
cudaMallocHost((void**)&C, size);  // Pinned memory for results
```

**Pinned Memory:**
- Allocated in main system memory (RAM)
- Cannot be moved by OS
- Faster GPU transfer than pageable memory
- Use `cudaMallocHost` instead of `malloc`

---

### Step 3: Initialize Host Data
```cpp
for(int i = 0; i < n; i++)
{
    A[i] = i;         // A = [0, 1, 2, ..., n-1]
    B[i] = i * 2;     // B = [0, 2, 4, ..., 2(n-1)]
}

// Expected result: C = A + B = [0, 3, 6, ..., 3(n-1)]
```

---

### Step 4: Allocate Device Memory (GPU)
```cpp
int *dev_A, *dev_B, *dev_C;  // Device pointers

cudaMalloc((void**)&dev_A, size);  // GPU memory for A
cudaMalloc((void**)&dev_B, size);  // GPU memory for B
cudaMalloc((void**)&dev_C, size);  // GPU memory for results
```

**Device Memory:**
- Located on GPU
- Much faster access for GPU threads
- Limited size (typically 2-24 GB)
- Must allocate and deallocate explicitly

---

### Step 5: Copy Data to GPU
```cpp
cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);  // Copy A to device
cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);  // Copy B to device
```

**cudaMemcpy parameters:**
- `dest`: Destination pointer (device)
- `src`: Source pointer (host)
- `size`: Bytes to copy
- `direction`: `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`

---

### Step 6: Configure and Launch Kernel
```cpp
int blockSize = 256;                           // Threads per block
int numBlocks = (n + blockSize - 1) / blockSize;  // Calculate needed blocks
```

**Grid configuration:**
- **blockSize = 256**: 256 threads per block (typical)
- **numBlocks calculation** (ceiling division):
  ```
  numBlocks = ceil(n / blockSize)
  
  For n=1,000,000, blockSize=256:
  numBlocks = (1,000,000 + 255) / 256 = 3907 blocks
  ```

**Launch kernel:**
```cpp
addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);
```

**Syntax:**
```
kernel<<<gridDim, blockDim>>>(args);
```

---

### Step 7: Copy Results Back to Host
```cpp
cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);  // Results to host
```

Direction: `cudaMemcpyDeviceToHost` (GPU → CPU)

---

### Step 8: Print Results
```cpp
cout << "First 10 Elements of Result Vector:\n";

for(int i = 0; i < 10; i++)
{
    cout << C[i] << " ";  // Should be [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
}

cout << endl;
```

---

### Step 9: Free GPU Memory
```cpp
cudaFree(dev_A);  // Free device vector A
cudaFree(dev_B);  // Free device vector B
cudaFree(dev_C);  // Free device vector C
```

---

### Step 10: Free Host Memory
```cpp
cudaFreeHost(A);  // Free pinned host memory A
cudaFreeHost(B);  // Free pinned host memory B
cudaFreeHost(C);  // Free pinned host memory C
```

---

## Part 2: Matrix Multiplication (ass4_2_matrix_mul.cu)

### Overview
Matrix multiplication using CUDA: C = A × B where all are N×N matrices

---

## Matrix Multiplication Kernel

```cpp
__global__ void matrixMultiply(int* A, int* B, int* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    
    if(row < N && col < N)                            // Boundary check
    {
        int sum = 0;
        
        for(int k = 0; k < N; k++)                    // Compute dot product
        {
            sum += A[row * N + k] * B[k * N + col];   // A[row][k] * B[k][col]
        }
        
        C[row * N + col] = sum;                       // C[row][col] = sum
    }
}
```

**2D Thread Indexing:**
- **blockIdx.x, blockIdx.y**: 2D block coordinates
- **threadIdx.x, threadIdx.y**: 2D thread coordinates in block
- **Global indices**: row and column in result matrix

**Row-major storage:**
```
Matrix C (N×N):
[0,0] [0,1] [0,2] ... [0,N-1]
[1,0] [1,1] [1,2] ... [1,N-1]
...

Linear array indexing: C[row * N + col]
```

**Dot product calculation:**
```
C[row][col] = Σ(k=0 to N-1) A[row][k] * B[k][col]
```

---

## Main Function for Matrix Multiplication

### Step 1: Define Matrix Size
```cpp
int N = 4;                    // 4×4 matrices
int size = N * N * sizeof(int);  // Total bytes for one matrix
```

---

### Step 2: Allocate and Initialize Host Matrices
```cpp
int *A, *B, *C;

cudaMallocHost((void**)&A, size);
cudaMallocHost((void**)&B, size);
cudaMallocHost((void**)&C, size);

// Initialize matrices
for(int i = 0; i < N * N; i++)
{
    A[i] = i + 1;  // A = [1, 2, 3, ..., 16]
    B[i] = i + 1;  // B = [1, 2, 3, ..., 16]
}
```

---

### Step 3: Allocate Device Matrices
```cpp
int *dev_A, *dev_B, *dev_C;

cudaMalloc((void**)&dev_A, size);
cudaMalloc((void**)&dev_B, size);
cudaMalloc((void**)&dev_C, size);
```

---

### Step 4: Copy Matrices to Device
```cpp
cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
```

---

### Step 5: Configure 2D Grid and Blocks
```cpp
dim3 threadsPerBlock(16, 16);  // 16×16 = 256 threads per block

dim3 numBlocks(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,  // X dimension
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y   // Y dimension
);
```

**Example for N=4:**
```
threadsPerBlock = (16, 16)
numBlocks = ((4 + 15) / 16, (4 + 15) / 16) = (1, 1)

Grid: 1×1 blocks
Each block: 16×16 threads (256 threads)
Only 4×4 threads do actual work, rest idle
```

**`dim3` struct:**
```cpp
dim3 grid(Gx, Gy, Gz);    // Grid dimensions
dim3 block(Bx, By, Bz);   // Block dimensions
```

---

### Step 6: Launch 2D Matrix Multiplication Kernel
```cpp
matrixMultiply<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
```

---

### Step 7: Copy Results Back
```cpp
cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
```

---

### Step 8: Print Result Matrix
```cpp
cout << "Result Matrix:\n";

for(int i = 0; i < N; i++)
{
    for(int j = 0; j < N; j++)
    {
        cout << C[i * N + j] << " ";
    }
    cout << endl;
}
```

---

### Step 9: Free Memory
```cpp
cudaFree(dev_A);
cudaFree(dev_B);
cudaFree(dev_C);

cudaFreeHost(A);
cudaFreeHost(B);
cudaFreeHost(C);
```

---

## Memory Transfer Diagram

```
HOST (CPU)              GPU
┌────────┐           ┌──────────┐
│ Vector │  Host→Dev │ Device   │
│ A      ├──────────►│ A        │
└────────┘           │          │
┌────────┐           │  GPU     │
│ Vector │  Host→Dev │ Memory   │
│ B      ├──────────►│ B        │
└────────┘           └──────────┘
                          │
                     Process kernel
                          │
┌────────┐           ┌──────────┐
│ Vector │  Dev→Host │ Device   │
│ C      │◄──────────│ C        │
└────────┘           └──────────┘
```

---

## Compilation & Execution

```bash
# Compile
nvcc ass4_1_vector_add.cu -o ass4_1_vector_add
nvcc ass4_2_matrix_mul.cu -o ass4_2_matrix_mul

# Run
./ass4_1_vector_add
./ass4_2_matrix_mul
```

---

## Key CUDA Concepts Summary

### 1. Thread Hierarchy
```
Grid (all threads launched)
├─ Block (group of threads)
│  ├─ Thread 0
│  ├─ Thread 1
│  └─ Thread N
├─ Block ...
└─ Block ...
```

### 2. Memory Types
| Type | Location | Speed | Size | Usage |
|------|----------|-------|------|-------|
| **Global** | GPU | Slow | Large | Main GPU memory |
| **Shared** | GPU | Fast | Limited | Shared within block |
| **Local** | GPU | Medium | Per-thread | Automatic variables |
| **Host** | CPU | Fast (for CPU) | Large | CPU side |
| **Pinned** | CPU | Faster transfer | Large | For GPU-CPU transfer |

### 3. Execution Model
- **Sequential on CPU**: Host code
- **Parallel on GPU**: Kernel code
- **Synchronization**: After kernel launch (kernel waits for completion)

---

## Performance Tips

1. **Block size**: Typically 256-512 threads
2. **Minimize memory transfers**: GPU time dominated by transfers
3. **Use pinned memory**: For frequent host-device transfers
4. **Coalesce memory access**: Threads access adjacent memory

---

## Error Checking (Optional Enhancement)

```cpp
cudaError_t err = cudaMalloc(&dev_A, size);
if(err != cudaSuccess) {
    cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
}

// After kernel launch
err = cudaGetLastError();
if(err != cudaSuccess) {
    cout << "Kernel error: " << cudaGetErrorString(err) << endl;
}
```
