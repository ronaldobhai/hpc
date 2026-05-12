# Assignment 2: Parallel Sorting Algorithms - Code Explanation

## Overview
This program implements two sorting algorithms (Bubble Sort and Merge Sort) in both sequential and parallel versions using OpenMP, demonstrating performance comparison between them.

---

## Helper Function

### isSorted() - Verification Function
```cpp
bool isSorted(vector<int>& arr) {
    for(int i = 1; i < arr.size(); i++)      // Iterate from second element
        if(arr[i] < arr[i - 1])               // Check if current < previous
            return false;                     // Not sorted if decreasing
    
    return true;                              // All elements in ascending order
}
```
**Purpose**: Validates that the array is properly sorted. Returns true if sorted in ascending order.

---

## Bubble Sort

### Sequential Bubble Sort
```cpp
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    
    for(int i = 0; i < n - 1; i++)            // Outer loop: n-1 passes
    {
        for(int j = 0; j < n - i - 1; j++)    // Inner loop: compare adjacent elements
        {
            if(arr[j] > arr[j + 1])           // If current > next
                swap(arr[j], arr[j + 1]);     // Swap them
        }
    }
}
```

**How it works**:
1. Compare adjacent elements
2. Swap if they're in wrong order
3. After each pass, largest element "bubbles" to end
4. Repeat for remaining elements

**Example**: [5, 2, 8, 1]
- Pass 1: [2, 5, 1, 8] (8 bubbles to end)
- Pass 2: [2, 1, 5, 8] (5 bubbles)
- Pass 3: [1, 2, 5, 8] (sorted)

**Time Complexity**: O(n²) - quadratic
**Space Complexity**: O(1) - in-place

---

### Parallel Bubble Sort
```cpp
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    
    for(int i = 0; i < n; i++)                // Main loop
    {
        if(i % 2 == 0)                        // Even iterations
        {
            #pragma omp parallel for          // Parallelize comparing even indices
            for(int j = 0; j < n - 1; j += 2) {
                if(arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
        else                                  // Odd iterations
        {
            #pragma omp parallel for          // Parallelize comparing odd indices
            for(int j = 1; j < n - 1; j += 2) {
                if(arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```

**Parallelization Strategy** (Odd-Even Sort):
1. **Even iteration**: Compare pairs (0,1), (2,3), (4,5), ... - independent, can parallelize
2. **Odd iteration**: Compare pairs (1,2), (3,4), (5,6), ... - independent, can parallelize
3. No race conditions because non-overlapping pairs

**Visualization**:
```
Even:  Compare (0-1), (2-3), (4-5)  [independent]
Odd:   Compare (1-2), (3-4), (5-6)  [independent]
```

**Time Complexity**: Still O(n²) but with parallelization speedup
**Speedup**: Better than sequential due to parallel iterations

---

## Merge Sort

### Merge Function - Helper for Merge Sort
```cpp
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;                       // Size of left subarray
    int n2 = r - m;                           // Size of right subarray
    
    vector<int> L(n1), R(n2);                 // Temporary arrays
    
    // Copy data to temporary arrays
    for(int i = 0; i < n1; i++)
        L[i] = arr[l + i];                    // Left half
    
    for(int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];                // Right half
    
    int i = 0, j = 0, k = l;                  // Pointers for merge
    
    // Merge the two sorted subarrays
    while(i < n1 && j < n2) {
        if(L[i] <= R[j])                      // If left element smaller
            arr[k++] = L[i++];                // Take from left
        else
            arr[k++] = R[j++];                // Take from right
    }
    
    // Copy remaining elements from left
    while(i < n1)
        arr[k++] = L[i++];
    
    // Copy remaining elements from right
    while(j < n2)
        arr[k++] = R[j++];
}
```

**Merge Process**: Combines two sorted subarrays into one sorted array
- **Time Complexity**: O(n) where n = total elements
- **Space Complexity**: O(n) for temporary arrays

---

### Sequential Merge Sort
```cpp
void sequentialMergeSort(vector<int>& arr, int l, int r) {
    if(l < r)                                 // Base case: more than one element
    {
        int m = l + (r - l) / 2;              // Find middle (avoid overflow)
        
        sequentialMergeSort(arr, l, m);       // Sort left half
        sequentialMergeSort(arr, m + 1, r);   // Sort right half
        
        merge(arr, l, m, r);                  // Merge sorted halves
    }
}
```

**How it works** (Divide & Conquer):
1. Divide array into two halves
2. Recursively sort each half
3. Merge the sorted halves

**Example**: [8, 3, 5, 4]
```
Split:    [8, 3] | [5, 4]
Sort:     [3, 8] | [4, 5]
Merge:    [3, 4, 5, 8]
```

**Time Complexity**: O(n log n) - much better than bubble sort
**Space Complexity**: O(n) - requires temporary arrays

---

### Parallel Merge Sort
```cpp
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if(l >= r)
        return;                               // Base case
    
    // Small subarrays use sequential version (avoid parallelization overhead)
    if(r - l < 1000) {
        sequentialMergeSort(arr, l, r);
        return;
    }
    
    int m = l + (r - l) / 2;                  // Find middle
    
    #pragma omp task shared(arr)              // Create task for left half
    parallelMergeSort(arr, l, m);
    
    #pragma omp task shared(arr)              // Create task for right half
    parallelMergeSort(arr, m + 1, r);
    
    #pragma omp taskwait                      // Wait for both tasks to complete
    
    merge(arr, l, m, r);                      // Merge sorted halves
}
```

**Parallelization Strategy**:
- `#pragma omp task`: Creates tasks for each recursive call
- `shared(arr)`: Array is shared among tasks
- `#pragma omp taskwait`: Synchronizes - waits for both subtasks before merging
- **Threshold (1000)**: Uses sequential sort for small arrays to reduce overhead

**Task Tree**:
```
        Sort[0,7]
       /          \
Sort[0,3]      Sort[4,7]
  /    \         /    \
S[0,1] S[2,3] S[4,5] S[6,7]
```

---

## Main Function Structure

```cpp
int main() {
    int n = 50000;                            // Array size
    vector<int> arr(n);
    
    srand(time(0));                           // Seed random number generator
    
    for(int i = 0; i < n; i++)
        arr[i] = rand() % 100000;             // Random integers 0-99999
    
    // Create 4 copies for testing
    vector<int> arr1 = arr;  // Sequential Bubble
    vector<int> arr2 = arr;  // Parallel Bubble
    vector<int> arr3 = arr;  // Sequential Merge
    vector<int> arr4 = arr;  // Parallel Merge
    
    int threads = omp_get_max_threads();      // Get number of available threads
    
    // ... Timing and execution code ...
}
```

---

## Timing and Performance Measurement

```cpp
double start, end;

start = omp_get_wtime();                      // Start timer
sequentialBubbleSort(arr1);
end = omp_get_wtime();

double seqBubble = end - start;               // Calculate duration
```

**Performance Metrics Calculated**:
```cpp
double bubbleSpeedup = seqBubble / parBubble;
double efficiency = bubbleSpeedup / threads;
double throughput = n / parMerge;  // elements per second
```

---

## Parallel Merge Sort Execution Pattern

```cpp
#pragma omp parallel
{
    #pragma omp single
    parallelMergeSort(arr4, 0, n - 1);  // Only one thread executes this
}
```

**Why `#pragma omp single`?**
- Ensures only one thread starts the initial task
- Other threads can work on created subtasks
- Prevents creating duplicate task trees

---

## Summary Comparison Table

| Feature | Sequential Bubble | Parallel Bubble | Sequential Merge | Parallel Merge |
|---------|---|---|---|---|
| **Algorithm** | Exchange sort | Odd-even sort | Divide & conquer | Task-based divide & conquer |
| **Time Complexity** | O(n²) | O(n²) | O(n log n) | O(n log n) |
| **Space Complexity** | O(1) | O(1) | O(n) | O(n) |
| **Parallelization** | None | Simple pairs | Recursive tasks | Recursive tasks |
| **Best Case** | Already sorted | Already sorted | O(n log n) | O(n log n) |
| **Worst Case** | Reverse sorted | Reverse sorted | O(n log n) | O(n log n) |
| **Stability** | Stable | Stable | Stable | Stable |
| **When to use** | Small arrays | Medium arrays | Large arrays | Very large arrays |

---

## Algorithm Characteristics

### Bubble Sort
- **Advantage**: Simple to implement and understand
- **Disadvantage**: Very slow for large datasets (O(n²))
- **Use when**: Educational purposes, small datasets only

### Merge Sort
- **Advantage**: Guaranteed O(n log n), stable, good for large data
- **Disadvantage**: Requires O(n) extra space
- **Use when**: Reliability matters, large datasets, external sorting

---

## Compilation & Execution

```bash
# Compile
g++ ass2_sorts.cpp -o ass2_sorts -fopenmp

# Run
./ass2_sorts
```

**Expected Output**: 
- Execution times for all four sorting methods
- Speedup and efficiency metrics
- Throughput in elements/second
- Verification that all sorts produced correct results
