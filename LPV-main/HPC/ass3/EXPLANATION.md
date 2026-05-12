# Assignment 3: Parallel Reduction Operations - Code Explanation

## Overview
This program implements reduction operations (sum, min, max, average) in both sequential and parallel versions using OpenMP, demonstrating performance improvements with parallel reduction.

---

## Parallel Reduction Concept

**Reduction**: Combining multiple values into a single result value.

**Examples:**
- Sum: 1 + 2 + 3 + 4 = 10
- Min: Find smallest of multiple numbers
- Max: Find largest of multiple numbers
- Average: Sum / count

---

## Sequential Operations Section

### Sequential Sum
```cpp
int sequentialSum(vector<int>& arr) {
    int sum = 0;                              // Accumulator
    
    for(int i = 0; i < arr.size(); i++)
        sum += arr[i];                        // Add each element
    
    return sum;
}
```

**How it works:**
1. Initialize sum to 0
2. Iterate through all elements
3. Add each element to sum
4. Return total sum

**Time Complexity**: O(n)

---

### Sequential Minimum
```cpp
int sequentialMin(vector<int>& arr) {
    int min_val = arr[0];                     // Start with first element
    
    for(int i = 1; i < arr.size(); i++) {
        if(arr[i] < min_val)                  // Compare with current minimum
            min_val = arr[i];                 // Update if smaller
    }
    
    return min_val;
}
```

**How it works:**
1. Initialize min_val to first element
2. Compare each subsequent element
3. Update min_val if smaller value found
4. Return minimum value

**Time Complexity**: O(n)

---

### Sequential Maximum
```cpp
int sequentialMax(vector<int>& arr) {
    int max_val = arr[0];                     // Start with first element
    
    for(int i = 1; i < arr.size(); i++) {
        if(arr[i] > max_val)                  // Compare with current maximum
            max_val = arr[i];                 // Update if larger
    }
    
    return max_val;
}
```

**How it works:**
1. Initialize max_val to first element
2. Compare each subsequent element
3. Update max_val if larger value found
4. Return maximum value

**Time Complexity**: O(n)

---

### Sequential Average
```cpp
double sequentialAvg(vector<int>& arr) {
    int sum = sequentialSum(arr);             // Get sum using sum function
    
    return (double)sum / arr.size();          // Divide by count for average
}
```

**How it works:**
1. Calculate sum of all elements
2. Divide by number of elements
3. Cast to double for precision
4. Return average

**Time Complexity**: O(n)

---

## Parallel Operations Section

### Parallel Sum with Reduction
```cpp
int parallelSum(vector<int>& arr) {
    int sum = 0;                              // Accumulator
    
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < arr.size(); i++)
        sum += arr[i];                        // Each thread adds to shared sum
    
    return sum;
}
```

**OpenMP Reduction Clause: `reduction(+:sum)`**
- **`+`**: Operator (can be: +, -, *, &, |, ^, &&, ||, min, max)
- **`sum`**: Variable being reduced

**How it works:**
1. Each thread gets a private copy of `sum` initialized to 0
2. Each thread adds elements to its private sum
3. At end of loop, all private sums are combined with `+` operator
4. Final result written back to original `sum`

**Parallel execution:**
```
Thread 0: sum = arr[0] + arr[1] + arr[2]
Thread 1: sum = arr[3] + arr[4] + arr[5]
Thread 2: sum = arr[6] + arr[7] + arr[8]
Thread 3: sum = arr[9] + arr[10] + arr[11]

Final sum = (Thread0 + Thread1 + Thread2 + Thread3)
```

---

### Parallel Minimum with Reduction
```cpp
int parallelMin(vector<int>& arr) {
    int min_val = arr[0];                     // Initialize to first element
    
    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < arr.size(); i++) {
        if(arr[i] < min_val)                  // Each thread compares
            min_val = arr[i];                 // Updates its copy
    }
    
    return min_val;
}
```

**OpenMP Reduction Clause: `reduction(min:min_val)`**
- Uses **`min`** operator to combine values
- Each thread finds local minimum
- All local minimums are compared to find global minimum

**How it works:**
1. Each thread gets private copy of min_val = arr[0]
2. Each thread finds minimum in its subset
3. At end, all private minimums are compared
4. Global minimum is the result

---

### Parallel Maximum with Reduction
```cpp
int parallelMax(vector<int>& arr) {
    int max_val = arr[0];                     // Initialize to first element
    
    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < arr.size(); i++) {
        if(arr[i] > max_val)                  // Each thread compares
            max_val = arr[i];                 // Updates its copy
    }
    
    return max_val;
}
```

**OpenMP Reduction Clause: `reduction(max:max_val)`**
- Uses **`max`** operator to combine values
- Each thread finds local maximum
- All local maximums are compared to find global maximum

---

### Parallel Average
```cpp
double parallelAvg(vector<int>& arr) {
    int sum = parallelSum(arr);               // Use parallel sum
    
    return (double)sum / arr.size();          // Calculate average
}
```

**Leverages parallel sum** for faster calculation.

---

## Main Function Execution Flow

```cpp
int main() {
    int n = 1000000;                          // 1 million elements
    
    vector<int> arr(n);
    
    srand(time(0));                           // Seed random generator
    
    for(int i = 0; i < n; i++)
        arr[i] = rand() % 10000;              // Random values 0-9999
    
    int threads = omp_get_max_threads();      // Get thread count
    
    // ... Timing and execution for all operations ...
}
```

---

## Timing Pattern

### Sum Operation Example
```cpp
double start, end;

// SEQUENTIAL SUM
start = omp_get_wtime();
int seqSum = sequentialSum(arr);
end = omp_get_wtime();
double seqSumTime = end - start;

// PARALLEL SUM
start = omp_get_wtime();
int parSum = parallelSum(arr);
end = omp_get_wtime();
double parSumTime = end - start;
```

This pattern is repeated for:
- Min operation
- Max operation
- Average operation

---

## Performance Metrics Calculation

```cpp
// SPEEDUP: How much faster is parallel version?
double speedup = seqSumTime / parSumTime;

// EFFICIENCY: How well are threads utilized?
double efficiency = speedup / threads;

// THROUGHPUT: Elements processed per second
double throughput = n / parSumTime;
```

**Example with 4 threads:**
- Sequential time: 0.1 seconds
- Parallel time: 0.03 seconds
- Speedup: 0.1 / 0.03 = 3.33x faster
- Efficiency: 3.33 / 4 = 83.25% (good!)
- Throughput: 1,000,000 / 0.03 = 33.3M elements/sec

---

## Reduction Operators Available in OpenMP

```cpp
#pragma omp parallel for reduction(OPERATOR:variable)
```

| Operator | Description | Initial Value |
|----------|---|---|
| `+` | Sum | 0 |
| `-` | Difference | 0 |
| `*` | Product | 1 |
| `&` | Bitwise AND | ~0 |
| `\|` | Bitwise OR | 0 |
| `^` | Bitwise XOR | 0 |
| `&&` | Logical AND | 1 |
| `\|\|` | Logical OR | 0 |
| `min` | Minimum | Largest value |
| `max` | Maximum | Smallest value |

---

## Data Set Size and Thread Count

```cpp
int n = 1000000;           // 1 million elements - realistic size
int threads = omp_get_max_threads();
```

**Why 1 million?**
- Large enough to show parallelization benefits
- Reduces timing noise
- Realistic for HPC applications
- Overhead becomes negligible

---

## Output Format

The program outputs:
1. **Dataset information**: Size and thread count
2. **Final results**: Sum, Min, Max, Average
3. **Execution times**: For each sequential and parallel operation
4. **Performance metrics**: Speedup, efficiency, throughput

---

## Compilation & Execution

```bash
# Compile
g++ ass3_parallel_reduction.cpp -o ass3_parallel_reduction -fopenmp

# Run
./ass3_parallel_reduction
```

---

## Why Parallel Reduction is Important

1. **Very common operation**: Many algorithms need to combine partial results
2. **Built-in optimization**: OpenMP handles synchronization efficiently
3. **No manual synchronization**: Avoids critical sections overhead
4. **Scalable**: Works automatically with any number of threads
5. **Error-free**: Prevents common race condition bugs

---

## Reduction vs Manual Parallelization

**Without reduction (race condition!):**
```cpp
int sum = 0;
#pragma omp parallel for
for(int i = 0; i < arr.size(); i++)
    sum += arr[i];  // WRONG! Multiple threads modify sum simultaneously
```

**With reduction (correct):**
```cpp
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < arr.size(); i++)
    sum += arr[i];  // CORRECT! Each thread has private copy
```

---

## Summary Table

| Operation | Sequential | Parallel | Speedup Factor |
|-----------|---|---|---|
| **Sum** | O(n) | O(n/p) | ~p (p = threads) |
| **Min** | O(n) | O(n/p) | ~p |
| **Max** | O(n) | O(n/p) | ~p |
| **Average** | O(n) | O(n/p) | ~p |

Where p = number of threads (typically 4-16)

---

## Important Notes

1. **Reduction is only for reduction operations** - not for general synchronization
2. **Private copies automatically created** - programmer doesn't manage them
3. **Automatic combining at end** - uses specified operator
4. **Data type matters** - must support the operator
5. **Memory overhead** - O(n_threads) extra memory for private copies
