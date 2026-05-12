# Assignment 2: Sorting Algorithms - Questions & Answers

## Q1: Why is merge sort faster than bubble sort?

**Answer:**
- **Bubble Sort**: O(n²) time complexity - quadratic growth
- **Merge Sort**: O(n log n) time complexity - logarithmic growth

For n=50,000:
- Bubble Sort: ~2.5 billion comparisons
- Merge Sort: ~830,000 comparisons
- Merge sort is about 3000x faster!

---

## Q2: What is "odd-even sort" in parallel bubble sort?

**Answer:**
Odd-even sort (also called brick sort) parallelizes bubble sort by:

**Even phase**: Compare and swap pairs at even indices
```
Array: [a, b, c, d, e, f]
Pairs: (0,1), (2,3), (4,5)  ← Can be done in parallel
```

**Odd phase**: Compare and swap pairs at odd indices
```
Array: [a, b, c, d, e, f]
Pairs: (1,2), (3,4), (5,6)  ← Can be done in parallel
```

**Advantage**: No conflicts since pairs are independent - no critical sections needed!

---

## Q3: How does the threshold (1000) help in parallel merge sort?

**Answer:**
```cpp
if(r - l < 1000) {
    sequentialMergeSort(arr, l, r);  // Use sequential for small arrays
    return;
}
```

**Why?**
- Creating tasks has overhead (context switching, scheduling)
- For small subarrays, sequential is faster
- Threshold = balance between parallelization benefit and overhead
- Typical range: 1000-10000 elements depending on system

---

## Q4: What would happen if we removed the taskwait?

**Answer:**
```cpp
#pragma omp taskwait  // Without this...
```

Race condition! The merge operation would start before both subtasks complete:
1. Task 1 is still sorting left half
2. Task 2 is still sorting right half
3. Merge tries to merge incomplete data
4. Produces incorrect, corrupted output

**`#pragma omp taskwait`** ensures synchronization before proceeding.

---

## Q5: Why do we create multiple copies of the array?

**Answer:**
```cpp
vector<int> arr1 = arr;  // Sequential Bubble
vector<int> arr2 = arr;  // Parallel Bubble
vector<int> arr3 = arr;  // Sequential Merge
vector<int> arr4 = arr;  // Parallel Merge
```

**Reasons:**
1. **Preserve original**: Keep unsorted copy for all tests
2. **Independent sorting**: Each algorithm sorts its own copy
3. **Fair comparison**: All algorithms start with identical data
4. **Different modifications**: Each sort modifies its array

---

## Q6: What is `omp_get_max_threads()`?

**Answer:**
Returns the maximum number of threads OpenMP can use:
```cpp
int threads = omp_get_max_threads();  // e.g., 4, 8, 16...
```

**Depends on:**
- System CPU cores
- OpenMP implementation
- `OMP_NUM_THREADS` environment variable

---

## Q7: Why is merge sort space complexity O(n)?

**Answer:**
```cpp
vector<int> L(n1), R(n2);  // Temporary arrays
```

**In merge function:**
- Creates temporary array for left half: O(n/2)
- Creates temporary array for right half: O(n/2)
- Total: O(n/2 + n/2) = O(n)

**Trade-off**: Extra space required but faster speed (O(n log n) vs O(n²))

---

## Q8: Can bubble sort be stable?

**Answer:**
**Yes, this implementation is stable.**

Stability: Equal elements maintain their original relative order.

```cpp
if(arr[j] > arr[j + 1])    // Note: NOT >=
    swap(arr[j], arr[j + 1]);
```

Using `>` (not `>=`) means equal elements don't swap - maintaining order.

---

## Q9: What is the difference between `#pragma omp parallel for` and `#pragma omp task`?

**Answer:**

| Feature | parallel for | task |
|---------|---|---|
| **Use case** | Loops with independent iterations | Recursive/hierarchical work |
| **Load balancing** | Built-in scheduling | Manual work distribution |
| **Overhead** | Low | Moderate |
| **Example** | Sorting pairs in odd-even | Recursive merge sort |
| **Synchronization** | Implicit at loop end | Requires `taskwait` |

---

## Q10: How would you measure speedup?

**Answer:**
```cpp
double seqTime = end_seq - start_seq;
double parTime = end_par - start_par;
double speedup = seqTime / parTime;

cout << "Speedup: " << speedup << "x faster";
```

**Interpretation:**
- Speedup = 4.0 → 4x faster with parallelization
- Speedup = 1.0 → No improvement
- Speedup < 1.0 → Slower (overhead > benefit)

**Ideal speedup** with 4 threads ≈ 4.0

---

## Q11: Why use `rand() % 100000` for random numbers?

**Answer:**
```cpp
arr[i] = rand() % 100000;  // Range: 0 to 99999
```

**Purpose:**
- Creates variety in test data
- `%` operator gives remainder
- Forces values into a limited range
- More realistic than sequential numbers

**Note**: Modern C++ would use `<random>` header for better randomness.

---

## Q12: What is the purpose of `isSorted()` function?

**Answer:**
Validates that the sorting was successful:
```cpp
bool isSorted(vector<int>& arr) {
    for(int i = 1; i < arr.size(); i++)
        if(arr[i] < arr[i - 1])
            return false;
    return true;
}
```

**Why important:**
- Ensures correct sorting (not just timing)
- Catches bugs in parallel implementation
- Verifies no data corruption
- Part of correctness testing

---

## Q13: Can you explain the merge function step by step?

**Answer:**
```cpp
void merge(vector<int>& arr, int l, int m, int r) {
    // Step 1: Calculate sizes
    int n1 = m - l + 1;  // Left subarray size
    int n2 = r - m;      // Right subarray size
    
    // Step 2: Create temporary arrays
    vector<int> L(n1), R(n2);
    
    // Step 3: Copy data into temporary arrays
    for(int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for(int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
    
    // Step 4: Merge process (two-pointer technique)
    int i = 0, j = 0, k = l;  // i for L, j for R, k for merged position
    
    while(i < n1 && j < n2) {
        if(L[i] <= R[j])
            arr[k++] = L[i++];  // Take from left if smaller
        else
            arr[k++] = R[j++];  // Take from right if smaller
    }
    
    // Step 5: Copy remaining elements
    while(i < n1)
        arr[k++] = L[i++];  // Remaining from left
    while(j < n2)
        arr[k++] = R[j++];  // Remaining from right
}
```

**Visualization**:
```
L = [2, 5]
R = [1, 8]
k=0, i=0, j=0

Compare: L[0]=2, R[0]=1 → Take R[0]=1
k=1, i=0, j=1

Compare: L[0]=2, R[1]=8 → Take L[0]=2
k=2, i=1, j=1

Compare: L[1]=5, R[1]=8 → Take L[1]=5
k=3, i=2, j=1

Copy remaining: R[1]=8
Result: [1, 2, 5, 8]
```

---

## Q14: What is efficiency and how is it calculated?

**Answer:**
```cpp
double efficiency = speedup / number_of_threads;
```

**Example with 4 threads:**
- Speedup = 3.8 → Efficiency = 3.8/4 = 0.95 (95%)
- Speedup = 2.0 → Efficiency = 2.0/4 = 0.50 (50%)

**Interpretation:**
- 1.0 (100%) = Perfect parallelization
- 0.5 (50%) = Threads are 50% utilized
- < 0.25 = Parallelization overhead dominates

---

## Q15: How does throughput measurement work?

**Answer:**
```cpp
double throughput = n / parMerge;  // n = number of elements
```

**Example:**
- n = 50,000 elements
- parMerge = 0.5 seconds
- Throughput = 50,000 / 0.5 = 100,000 elements/second

**Meaning**: Algorithm processes 100,000 elements per second

---

## Q16: Why use bit shifting `r - l < 1000` instead of exact value?

**Answer:**
The threshold of 1000 is empirically chosen:
- Too small: Creates too many tasks, overhead dominates
- Too large: Doesn't parallelize small enough subproblems
- 1000 is a heuristic for typical systems

**Could be optimized by:**
```cpp
// Adaptive threshold based on hardware
int threshold = (int)pow(2, log2(n) - 2);  // n/4
if(r - l < threshold)
    sequentialMergeSort(arr, l, r);
```

---

## Q17: Can parallel bubble sort be stable?

**Answer:**
**Yes**, because of how odd-even sort works:
- Non-adjacent pairs swap independently
- Equal elements are never compared
- Relative order maintained

```cpp
if(arr[j] > arr[j + 1])  // Not >= , so stable
    swap(arr[j], arr[j + 1]);
```

---

## Q18: What happens with already sorted data?

**Answer:**
- **Bubble Sort**: Still O(n²) - no optimization
- **Merge Sort**: Still O(n log n) - algorithm doesn't check if sorted
- Neither has optimization for pre-sorted data

**Optimized bubble sort** could use:
```cpp
bool swapped = false;
for(int i = 0; i < n - 1; i++) {
    swapped = false;
    for(int j = 0; j < n - i - 1; j++) {
        if(arr[j] > arr[j + 1]) {
            swap(arr[j], arr[j + 1]);
            swapped = true;
        }
    }
    if(!swapped) break;  // Exit if no swaps
}
```

---

## Q19: How would you implement quicksort with OpenMP?

**Answer:**
```cpp
void quickSort(vector<int>& arr, int l, int r) {
    if(l >= r) return;
    
    int p = partition(arr, l, r);  // Partition around pivot
    
    #pragma omp task shared(arr)
    quickSort(arr, l, p - 1);
    
    #pragma omp task shared(arr)
    quickSort(arr, p + 1, r);
    
    #pragma omp taskwait
}

// In main:
#pragma omp parallel
{
    #pragma omp single
    quickSort(arr, 0, n - 1);
}
```

**Advantages**: Better cache locality than merge sort
**Disadvantage**: Unstable, O(n²) worst case

---

## Q20: What is the best sorting algorithm to use in practice?

**Answer:**

| Scenario | Best Algorithm | Reason |
|----------|---|---|
| Small data (< 100k) | Insertion sort | Fast for tiny data |
| Medium data (100k-1M) | Merge sort | Reliable O(n log n) |
| Large data (> 1M) | Parallel merge sort | Leverages multiple cores |
| Nearly sorted data | Insertion sort | Optimized for pre-sorted |
| Limited memory | Quicksort | Only O(log n) extra space |
| Need stability | Merge sort | Guaranteed stable |
| Need speed (practical) | std::sort (Introsort) | Combines quick+heap+insertion |
| Parallel processing | Parallel merge sort | O(n log n) with speedup |

**Modern C++ Standard**:
```cpp
#include <algorithm>
std::sort(arr.begin(), arr.end());  // Uses Introsort - best of all worlds
```
