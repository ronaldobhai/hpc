# Assignment 4: CUDA Programming - Questions & Answers

## Q1: What does `__global__` mean in CUDA?

**Answer:**
`__global__` is a CUDA function qualifier that:
- Runs on **GPU** (device)
- Called from **CPU** (host)
- Executed by many threads in parallel

**Three CUDA function types:**
```cpp
__global__    // CPU calls → Runs on GPU
__device__    // GPU only → Only called from GPU
__host__      // CPU only → Only called from CPU (default)
```

**Example:**
```cpp
__global__ void kernel() { }      // GPU kernel
__device__ int helper() { }       // GPU helper
__host__ int main() { }           // CPU main
```

---

## Q2: How is the global thread index calculated?

**Answer:**
For 1D kernel:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**Breakdown:**
- **blockIdx.x**: Which block (0, 1, 2, ...)
- **blockDim.x**: Threads per block
- **threadIdx.x**: Position within block (0 to blockDim.x-1)

**Example:**
```
Block 0: Threads 0-255 (threadIdx.x)
Block 1: Threads 0-255 (threadIdx.x)

blockIdx.x=1, blockDim.x=256, threadIdx.x=5
Global index = (1 × 256) + 5 = 261
```

**Visualization:**
```
Global index: 0   1   2   ... 255 | 256 257 258 ... 511 |
              └─ Block 0 ────────┘  └─ Block 1 ────────┘
```

---

## Q3: Why use pinned memory?

**Answer:**
```cpp
cudaMallocHost(&A, size);  // Pinned memory
```

**Pinned (Page-locked) Memory:**
1. **Faster transfers**: CPU-GPU transfers ~2x faster
2. **Can't swap**: OS can't move to disk
3. **Direct DMA**: GPU can access directly
4. **Limited size**: Can't allocate too much

**Comparison:**
```
Pageable memory (malloc):  ~5 GB/s transfer
Pinned memory (cudaMallocHost): ~11 GB/s transfer
```

**Tradeoff:**
- Faster GPU transfer
- Limited available pinned memory
- More restricted than regular malloc

---

## Q4: What is the purpose of `if(i < n)` in the kernel?

**Answer:**
**Boundary check** to prevent out-of-bounds access.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < n)  // Don't access array beyond n
{
    C[i] = A[i] + B[i];
}
```

**Why needed:**
```
n = 1,000,000 (not perfectly divisible by 256)
blockSize = 256
numBlocks = (1,000,000 + 255) / 256 = 3907 blocks

Total threads created: 3907 × 256 = 1,000,192
Actual elements: 1,000,000
Extra threads: 192 (would access invalid memory!)
```

**Without check:**
```cpp
C[1,000,100] = A[1,000,100] + B[1,000,100];  // Out of bounds! Crash or corruption!
```

**With check:**
```cpp
if(1,000,100 < 1,000,000) { ... }  // false, skip - safe!
```

---

## Q5: What is `dim3` in CUDA?

**Answer:**
`dim3` is a structure for 3D dimensions:
```cpp
struct dim3 {
    unsigned int x, y, z;
};
```

**Usage:**
```cpp
dim3 threadsPerBlock(16, 16, 1);   // 16×16×1 = 256 threads
dim3 numBlocks(4, 4, 1);           // 4×4×1 = 16 blocks

kernel<<<numBlocks, threadsPerBlock>>>();
```

**Accessing in kernel:**
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int depth = blockIdx.z * blockDim.z + threadIdx.z;
```

**Default values:**
- If not specified, .y and .z default to 1
- `dim3 grid(4)` = `dim3(4, 1, 1)`

---

## Q6: How do you calculate the correct number of blocks?

**Answer:**
**Ceiling division** to ensure all elements are covered:
```cpp
int numBlocks = (n + blockSize - 1) / blockSize;
```

**Why the formula?**
```
Regular division (wrong):
  n=10, blockSize=3 → 10/3 = 3 blocks → Only 9 threads (not enough!)

Ceiling division (correct):
  n=10, blockSize=3 → (10+2)/3 = 4 blocks → 12 threads (covers all)
```

**Examples:**
```
n=100, blockSize=32:
  numBlocks = (100 + 31) / 32 = 131 / 32 = 4 blocks
  Total threads = 4 × 32 = 128 ✓

n=1000, blockSize=256:
  numBlocks = (1000 + 255) / 256 = 1255 / 256 = 4 blocks
  Total threads = 4 × 256 = 1024 ✓
```

---

## Q7: What does `cudaMemcpy` do?

**Answer:**
Copies data between host and device memory.

**Syntax:**
```cpp
cudaMemcpy(destination, source, size, direction);
```

**Directions:**
```cpp
cudaMemcpyHostToDevice      // CPU → GPU
cudaMemcpyDeviceToHost      // GPU → CPU
cudaMemcpyDeviceToDevice    // GPU → GPU
cudaMemcpyHostToHost        // CPU → CPU (rarely used)
```

**Examples:**
```cpp
// Copy to device before kernel
cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);

// Copy from device after kernel
cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
```

---

## Q8: What is the difference between `cudaMalloc` and `cudaMallocHost`?

**Answer:**

| Feature | cudaMalloc | cudaMallocHost |
|---------|---|---|
| **Location** | GPU memory | CPU memory (pinned) |
| **Access** | GPU threads | Both CPU and GPU |
| **Speed** | Fast for GPU | Fast for transfers |
| **Size** | Limited (2-24 GB) | Large (100s GB) |
| **Free with** | cudaFree | cudaFreeHost |
| **Use for** | Data on GPU | Data transfer |

**Typical usage:**
```cpp
int *A, *dev_A;

cudaMallocHost(&A, size);     // Host: pinned
cudaMalloc(&dev_A, size);     // Device: GPU

cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);  // Transfer
```

---

## Q9: How does matrix multiplication work in CUDA?

**Answer:**
Each thread computes one element of result matrix C.

**C = A × B computation:**
```
C[row][col] = Σ(k=0 to N-1) A[row][k] * B[k][col]
```

**Example:**
```
A = │1 2│    B = │1 0│    C = │1×1+2×2  1×0+2×3│ = │5  6│
    │3 4│        │2 3│        │3×1+4×2  3×0+4×3│   │11 12│

C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*1 + 2*2 = 5
C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] = 1*0 + 2*3 = 6
C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] = 3*1 + 4*2 = 11
C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] = 3*0 + 4*3 = 12
```

**Kernel logic:**
```cpp
__global__ void matrixMultiply(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        int sum = 0;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

---

## Q10: What is row-major storage?

**Answer:**
2D data stored sequentially in 1D array row by row.

**2D Matrix:**
```
│1  2  3│
│4  5  6│
│7  8  9│
```

**Row-major (C/C++ standard):**
```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Indexing:**
```
matrix[i][j] → array[i * N + j]

matrix[0][2] → array[0*3 + 2] = array[2] = 3 ✓
matrix[1][1] → array[1*3 + 1] = array[4] = 5 ✓
matrix[2][0] → array[2*3 + 0] = array[6] = 7 ✓
```

**Column-major (Fortran/MATLAB):**
```
[1, 4, 7, 2, 5, 8, 3, 6, 9]

Indexing: matrix[i][j] → array[j * N + i]
```

---

## Q11: What happens if you forget `cudaFree`?

**Answer:**
**Memory leak!** GPU memory not released.

```cpp
cudaMalloc(&dev_A, size);
// ... use dev_A ...
// Forgot: cudaFree(dev_A);

// Memory stays allocated until program ends!
```

**Consequences:**
- Available GPU memory decreases
- Eventually can't allocate more
- Other programs suffer
- Only cleared when GPU resets

**Proper cleanup:**
```cpp
// Allocate
int *dev_A, *dev_B, *dev_C;
cudaMalloc(&dev_A, size);
cudaMalloc(&dev_B, size);
cudaMalloc(&dev_C, size);

// Use...

// Cleanup
cudaFree(dev_A);
cudaFree(dev_B);
cudaFree(dev_C);
```

---

## Q12: Can you mix CPU and GPU code?

**Answer:**
**Yes!** That's the whole point of CUDA.

**Typical pattern:**
```cpp
int *A, *dev_A;

// 1. Host code: Allocate
cudaMallocHost(&A, size);
cudaMalloc(&dev_A, size);

// 2. Host code: Initialize
for(int i = 0; i < n; i++)
    A[i] = rand();

// 3. Host code: Transfer
cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);

// 4. Device code: Compute (in kernel)
// kernel<<<blocks, threads>>>(dev_A);

// 5. Host code: Transfer back
cudaMemcpy(A, dev_A, size, cudaMemcpyDeviceToHost);

// 6. Host code: Process results
for(int i = 0; i < n; i++)
    cout << A[i];

// 7. Host code: Cleanup
cudaFree(dev_A);
cudaFreeHost(A);
```

---

## Q13: What is warp and warp divergence?

**Answer:**
**Warp**: 32 threads executed together as a unit.

**NVIDIA GPUs:**
- Execute 32 threads in lockstep
- All threads run same instruction simultaneously
- Can't be independent within a warp

**Warp divergence (control flow issue):**
```cpp
if(condition) {
    // Path A
} else {
    // Path B
}
```

**Problem:**
- Some threads in warp take Path A
- Others take Path B
- GPU must execute **both paths sequentially**
- Threads taking different path are idle
- Performance penalty!

**Example:**
```cpp
if(threadIdx.x % 2 == 0) {
    // 16 threads per warp do work
    result = compute1();
} else {
    // 16 threads wait, then do work
    result = compute2();
}
// Effective throughput = 50%
```

**Best practice:**
- Minimize branching in kernels
- Keep threads in a warp doing similar work

---

## Q14: How many threads should a block have?

**Answer:**
**Typical: 128-512 threads per block**

**Factors:**
1. **GPU generation**: Older = smaller, newer = larger
2. **Shared memory**: Limited resource, reduces if threads increase
3. **Occupancy**: Percentage of GPU utilized

**Common choices:**
```
32 threads:   Low occupancy, simple kernels
64 threads:   Good balance
128 threads:  Common choice
256 threads:  Very common (default)
512 threads:  High resource demand
1024 threads: Maximum per block
```

**Example optimization:**
```cpp
// Find optimal block size
for(int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Measure performance
    kernel<<<numBlocks, blockSize>>>();
    
    // Record time
}
```

---

## Q15: What is occupancy and why does it matter?

**Answer:**
**Occupancy**: Percentage of maximum possible threads running on GPU.

```
Occupancy = (Active threads) / (Maximum possible threads) × 100%
```

**Example:**
```
GPU max threads: 2048 per block
Block size: 256 threads
Occupancy per block: 256/2048 = 12.5%

If we have 8 blocks:
Total threads: 8 × 256 = 2048
Occupancy: 100%
```

**Why important:**
- **High occupancy**: Better GPU utilization
- **Low occupancy**: Idle GPU hardware, wasted potential
- **Memory latency hiding**: High occupancy hides memory delays

**Improving occupancy:**
1. Reduce shared memory usage
2. Increase block size
3. Use more blocks

---

## Q16: How does synchronization work in CUDA?

**Answer:**
**Within block:**
```cpp
__syncthreads();  // All threads in block wait here
```

**Between kernels:**
```cpp
kernel1<<<blocks, threads>>>();
// GPU waits for kernel1 to complete
kernel2<<<blocks, threads>>>();  // Starts after kernel1
```

**Host-device sync:**
```cpp
kernel<<<blocks, threads>>>();
cudaMemcpy(...);  // Implicit synchronization
```

---

## Q17: Can you call device functions from device code?

**Answer:**
**Yes!** Using `__device__` qualifier.

```cpp
__device__ int add(int a, int b) {
    return a + b;
}

__global__ void kernel() {
    int result = add(5, 3);  // Call device function
}
```

**Cannot call from host:**
```cpp
int main() {
    add(5, 3);  // ERROR! Can't call __device__ from host
}
```

---

## Q18: What is the block limit per grid?

**Answer:**
**Modern GPUs: Virtually unlimited blocks**

**Theoretically:**
- Grid can have up to 2^31 - 1 blocks in each dimension (x, y, z)
- Practical limit: GPU memory and launch overhead

**Example:**
```
1D grid:  Up to 2^31 blocks
2D grid:  Up to 2^16 × 2^16 blocks
3D grid:  Up to 2^10 × 2^10 × 2^10 blocks
```

This allows handling arbitrarily large problems by creating more blocks.

---

## Q19: How to query GPU properties?

**Answer:**
```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
cout << "Number of GPUs: " << deviceCount << endl;

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // GPU 0

cout << "GPU Name: " << prop.name << endl;
cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
cout << "Max blocks per dim: " << prop.maxGridSize[0] << endl;
cout << "Warp size: " << prop.warpSize << endl;
cout << "Shared memory per block: " << prop.sharedMemPerBlock << endl;
```

---

## Q20: Vector addition vs Matrix multiplication efficiency?

**Answer:**

**Vector Addition:**
- **Computation**: O(1) per element
- **Memory**: Load 2, store 1 (3 operations)
- **Computation/Memory ratio**: Low
- **Efficiency**: Memory bound

**Matrix Multiplication:**
- **Computation**: O(N) per element
- **Memory**: Load 2N, store 1 (2N+1 operations)
- **Computation/Memory ratio**: High
- **Efficiency**: Compute bound

**GPU is better suited for:**
- Matrix multiplication (more computation)
- Complex algorithms with high compute intensity

**Vector addition:**
- GPU overhead often exceeds benefit
- Better on multi-core CPU with SIMD
- Good for learning CUDA
