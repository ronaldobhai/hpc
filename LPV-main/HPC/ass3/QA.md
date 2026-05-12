# Assignment 3: Parallel Reduction - Questions & Answers

## Q1: What is reduction in parallel computing?

**Answer:**
Reduction is an operation that combines multiple values into a single result using an associative operator.

**Examples:**
- Sum: 1 + 2 + 3 + 4 = 10
- Min: min(5, 2, 8, 1) = 1
- Max: max(5, 2, 8, 1) = 8
- Product: 2 × 3 × 4 = 24

**Key property**: Associative - order of combining intermediate results doesn't matter:
```
(a + b) + c = a + (b + c)
(a * b) * c = a * (b * c)
```

---

## Q2: How does `#pragma omp parallel for reduction(+:sum)` work?

**Answer:**
```cpp
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < arr.size(); i++)
    sum += arr[i];
```

**Step-by-step execution:**

1. **Before loop**: Each thread creates a **private copy** of `sum` initialized to 0
   ```
   Thread 0: sum = 0 (private)
   Thread 1: sum = 0 (private)
   Thread 2: sum = 0 (private)
   Thread 3: sum = 0 (private)
   ```

2. **During loop**: Each thread adds to **its own private copy**
   ```
   Thread 0: sum += arr[0], arr[1], arr[2]     → sum = 3
   Thread 1: sum += arr[3], arr[4], arr[5]     → sum = 12
   Thread 2: sum += arr[6], arr[7], arr[8]     → sum = 21
   Thread 3: sum += arr[9], arr[10], arr[11]   → sum = 30
   ```

3. **After loop**: OpenMP **combines** all private sums using `+` operator
   ```
   sum = 3 + 12 + 21 + 30 = 66 (final result)
   ```

**No race condition!** Each thread has its own copy.

---

## Q3: What happens without the reduction clause?

**Answer:**
**Race condition and incorrect results!**

```cpp
int sum = 0;
#pragma omp parallel for
for(int i = 0; i < arr.size(); i++)
    sum += arr[i];  // WRONG!
```

**Why it fails:**
1. Multiple threads access same variable `sum`
2. Without reduction, `sum` is **shared** (not private)
3. Threads overwrite each other's updates
4. Result is unpredictable and usually wrong

**Example:**
```
Thread 0: reads sum=0, adds arr[0]=1, writes sum=1
Thread 1: reads sum=0, adds arr[1]=2, writes sum=2  (lost Thread 0's work!)
Thread 2: reads sum=2, adds arr[2]=3, writes sum=5
Final sum might be 5 instead of correct answer!
```

---

## Q4: What are valid reduction operators?

**Answer:**
OpenMP supports these reduction operators:

**Arithmetic:**
- `+` (sum) - init: 0
- `-` (subtraction) - init: 0
- `*` (product) - init: 1

**Bitwise:**
- `&` (AND) - init: ~0 (all 1s)
- `|` (OR) - init: 0
- `^` (XOR) - init: 0

**Logical:**
- `&&` (AND) - init: 1
- `||` (OR) - init: 0

**Comparison:**
- `min` (minimum) - init: max value
- `max` (maximum) - init: min value

---

## Q5: What are the initial values for reduction variables?

**Answer:**
OpenMP automatically initializes private copies based on the operator:

| Operator | Initial Value | Reason |
|----------|---|---|
| `+` | 0 | Identity for addition |
| `-` | 0 | Identity for subtraction |
| `*` | 1 | Identity for multiplication |
| `&` | ~0 (all 1s) | Identity for bitwise AND |
| `\|` | 0 | Identity for bitwise OR |
| `^` | 0 | Identity for bitwise XOR |
| `&&` | 1 (true) | Identity for logical AND |
| `\|\|` | 0 (false) | Identity for logical OR |
| `min` | INT_MAX | Largest possible value |
| `max` | INT_MIN | Smallest possible value |

---

## Q6: Why is the `min` reduction initialized to the maximum value?

**Answer:**
To correctly find the minimum value!

```cpp
int min_val = arr[0];  // If not set to INT_MAX...
#pragma omp parallel for reduction(min:min_val)
for(int i = 0; i < arr.size(); i++) {
    if(arr[i] < min_val)
        min_val = arr[i];
}
```

If initialized to 0:
- If all values are positive, might keep private min as 0
- Wrong result!

If initialized to INT_MAX:
- Any value compared with INT_MAX is smaller
- Correctly finds actual minimum

**Example with wrong init (0):**
```
Array: [5, 10, 8]
If init to 0:
  Thread 0: min=0, checks 5 (5 < 0? No), keeps 0
  Result: 0 (WRONG!)

If init to INT_MAX:
  Thread 0: min=INT_MAX, checks 5 (5 < INT_MAX? Yes), min=5
  Result: 5 (CORRECT!)
```

---

## Q7: How is speedup calculated?

**Answer:**
```cpp
double speedup = sequentialTime / parallelTime;
```

**Example:**
- Sequential execution: 1.0 second
- Parallel execution: 0.25 second
- Speedup: 1.0 / 0.25 = 4.0

**Interpretation:**
- Speedup = 4.0 means **4x faster** with parallelization
- Perfect speedup with 4 threads = 4.0
- Speedup > 4.0 = Superlinear (rare, due to cache effects)
- Speedup < 1.0 = Parallelization slower (overhead > benefit)

---

## Q8: What is efficiency and why does it matter?

**Answer:**
```cpp
double efficiency = speedup / numberOfThreads;
```

**Example with 4 threads:**
- Speedup = 3.8
- Efficiency = 3.8 / 4 = 0.95 = **95%**

**Interpretation:**
- 95% efficiency = Threads are 95% utilized
- 100% efficiency = Perfect parallelization
- 25% efficiency = Threads are 25% utilized (3/4 wasted)

**Why important:**
- Shows if adding more threads would help
- High efficiency = Good parallelization
- Low efficiency = Bottleneck or overhead issue

---

## Q9: Why does parallel reduction need private copies?

**Answer:**
To **avoid race conditions** and allow **independent computation**.

**Without private copies:**
```cpp
int sum = 0;  // Shared by all threads
#pragma omp parallel for
for(...) {
    sum += arr[i];  // Multiple threads read-modify-write
}
```

**Race condition:**
```
Time T0: Thread A reads sum=0
Time T1: Thread B reads sum=0
Time T2: Thread A writes sum=5 (after adding arr[i])
Time T3: Thread B writes sum=3 (overwrites A's value!)
Result: Only B's work counted
```

**With private copies:**
```
Thread A: private_sum_A += elements in its range
Thread B: private_sum_B += elements in its range
Thread C: private_sum_C += elements in its range
Thread D: private_sum_D += elements in its range

Final: sum = private_sum_A + private_sum_B + private_sum_C + private_sum_D
```

No conflicts!

---

## Q10: What is throughput and how is it calculated?

**Answer:**
```cpp
double throughput = n / parallelTime;  // elements per second
```

**Example:**
- Dataset size: 1,000,000 elements
- Parallel execution time: 0.01 seconds
- Throughput: 1,000,000 / 0.01 = **100 million elements/second**

**Use case:** Comparing algorithm performance:
- Algorithm A: 50M elements/sec
- Algorithm B: 100M elements/sec
- Algorithm B is twice as fast

---

## Q11: Can you parallelize average calculation differently?

**Answer:**
```cpp
// Current approach: calculate sum, then divide
double parallelAvg(vector<int>& arr) {
    int sum = parallelSum(arr);
    return (double)sum / arr.size();
}

// Alternative: calculate sum and count in parallel
double parallelAvgV2(vector<int>& arr) {
    int sum = 0;
    int count = 0;
    
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    
    return (double)sum / arr.size();  // count is always arr.size()
}

// More complex: what if some elements are invalid?
double parallelAvgFiltered(vector<int>& arr, int threshold) {
    int sum = 0;
    int count = 0;
    
    #pragma omp parallel for reduction(+:sum,count)
    for(int i = 0; i < arr.size(); i++) {
        if(arr[i] >= threshold) {
            sum += arr[i];
            count++;
        }
    }
    
    return count > 0 ? (double)sum / count : 0;
}
```

The `reduction(+:sum,count)` syntax allows reducing multiple variables!

---

## Q12: What data types work with reduction?

**Answer:**
Built-in numeric types:
- `int`, `float`, `double`, `long`, `long long`
- `unsigned int`, `unsigned long`, etc.

**Operators and compatible types:**

| Operator | Compatible Types |
|----------|---|
| `+`, `-` | All numeric |
| `*` | All numeric |
| `&`, `\|`, `^` | Integral types only |
| `&&`, `\|\|` | All types (converted to bool) |
| `min`, `max` | All numeric |

**User-defined types:**
- Not directly supported by OpenMP
- Must implement for built-in types
- Can create custom reduction with `#pragma omp declare reduction`

---

## Q13: How does reduction affect memory usage?

**Answer:**
Memory overhead = `size of variable × number_of_threads`

**Example:**
```cpp
int sum;  // 4 bytes
#pragma omp parallel for reduction(+:sum)
```

With 4 threads:
- Original variable: 4 bytes
- Private copies: 4 × 4 = 16 bytes
- Total: ~20 bytes extra

**Worst case:**
```cpp
vector<int> arr;  // 1M elements = 4MB
int sum;
#pragma omp parallel for reduction(+:sum)
// Only sum uses extra memory, not arr!
```

**With large reduction variables:**
```cpp
struct LargeData { int arr[1000]; };
LargeData data;
#pragma omp parallel for reduction(+:data.arr[0])
// Only the one int uses extra memory
```

---

## Q14: What happens with nested parallelism and reduction?

**Answer:**
```cpp
#pragma omp parallel
{
    int local_sum = 0;
    #pragma omp for reduction(+:local_sum)
    for(int i = 0; i < n; i++) {
        local_sum += arr[i];
    }
    // local_sum is now combined
}
```

**Nested example (problematic):**
```cpp
#pragma omp parallel reduction(+:sum)  // Outer reduction
{
    #pragma omp parallel reduction(+:sum)  // Inner reduction
    {
        sum += compute();  // Both try to reduce!
    }
}
// May cause unexpected behavior
```

**Better approach:**
```cpp
int global_sum = 0;

#pragma omp parallel
{
    int thread_sum = 0;
    
    #pragma omp for
    for(int i = 0; i < n; i++) {
        thread_sum += arr[i];
    }
    
    #pragma omp critical
    {
        global_sum += thread_sum;
    }
}
```

---

## Q15: Can reduction work with other OpenMP constructs?

**Answer:**
Yes! Reduction works with:

**`parallel for` (most common):**
```cpp
#pragma omp parallel for reduction(+:sum)
for(...) { sum += ...; }
```

**`parallel sections`:**
```cpp
#pragma omp parallel sections reduction(+:sum)
{
    #pragma omp section
    { sum += computeA(); }
    
    #pragma omp section
    { sum += computeB(); }
}
```

**`parallel` + `for`:**
```cpp
#pragma omp parallel reduction(+:sum)
{
    #pragma omp for
    for(...) { sum += ...; }
}
```

---

## Q16: When would you use `reduction` vs `atomic`?

**Answer:**

| Feature | Reduction | Atomic |
|---------|---|---|
| **Syntax** | `reduction(op:var)` | `#pragma omp atomic` |
| **Use case** | Aggregating values | Single update |
| **Performance** | Fast (no synchronization in loop) | Slower (sync at each update) |
| **Code** | Cleaner | More verbose |
| **Flexibility** | Predefined operators | Any operation |

**Reduction (preferred for aggregation):**
```cpp
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < n; i++)
    sum += arr[i];
```

**Atomic (for custom operations):**
```cpp
int count = 0;
#pragma omp parallel for
for(int i = 0; i < n; i++) {
    if(arr[i] % 2 == 0) {
        #pragma omp atomic
        count++;  // Custom condition
    }
}
```

---

## Q17: How would you reduce a 2D array?

**Answer:**
```cpp
int matrix[10][10];
int sum = 0;

#pragma omp parallel for collapse(2) reduction(+:sum)
for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
        sum += matrix[i][j];
    }
}
```

**`collapse(2)`** parallelizes both loops:
- Flattens nested loops
- Distributes iterations among threads
- Combines with reduction

**Alternative (single loop):**
```cpp
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < 100; i++) {
    int row = i / 10;
    int col = i % 10;
    sum += matrix[row][col];
}
```

---

## Q18: What are associativity requirements for reduction?

**Answer:**
Associativity means the order of combining results doesn't matter:

**Associative (valid for reduction):**
```
(a + b) + c = a + (b + c)           ✓
(a * b) * c = a * (b * c)           ✓
(a & b) & c = a & (b & c)           ✓
(a | b) | c = a | (b | c)           ✓
max(max(a,b),c) = max(a,max(b,c))   ✓
min(min(a,b),c) = min(a,min(b,c))   ✓
```

**Non-associative (NOT valid for reduction):**
```
(a - b) - c ≠ a - (b - c)           ✗
(a / b) / c ≠ a / (b / c)           ✗
```

With threads, combining order is unpredictable:
- Thread 0 partial result: r0
- Thread 1 partial result: r1
- Thread 2 partial result: r2

Could combine as: (r0 + r1) + r2 or (r0 + r2) + r1 or r0 + (r1 + r2)

For non-associative operations, results differ!

---

## Q19: How would you profile reduction performance?

**Answer:**
```cpp
#include <omp.h>

// Measure times for different thread counts
for(int num_threads = 1; num_threads <= 8; num_threads *= 2) {
    omp_set_num_threads(num_threads);
    
    double start = omp_get_wtime();
    
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += arr[i];
    
    double end = omp_get_wtime();
    
    double time = end - start;
    double speedup = baseline_time / time;
    
    printf("Threads: %d, Time: %.6f, Speedup: %.2f\n", 
           num_threads, time, speedup);
}
```

**Output might show:**
```
Threads: 1, Time: 1.000000, Speedup: 1.00
Threads: 2, Time: 0.520000, Speedup: 1.92
Threads: 4, Time: 0.270000, Speedup: 3.70
Threads: 8, Time: 0.160000, Speedup: 6.25
```

Diminishing returns with more threads due to overhead.

---

## Q20: When would reduction be slower than sequential execution?

**Answer:**

1. **Small dataset:**
   ```cpp
   #pragma omp parallel for reduction(+:sum)
   for(int i = 0; i < 10; i++)  // Too small!
       sum += arr[i];
   ```
   Overhead of thread creation > benefit

2. **Complex reduction:**
   ```cpp
   struct Complex { 
       int a[1000], b[1000], c[1000]; 
   };
   // Large private copies for each thread!
   ```

3. **Already optimized sequential code:**
   - Cache-friendly, vectorized, compiled well
   - Hard to beat with naive parallel version

4. **Single-threaded system:**
   - Hardware has only 1 core
   - Parallelization overhead wasted

**Solution:**
- Only parallelize if benefit > overhead
- Use larger datasets
- Profile to measure improvement
- Consider `if` conditions for automatic fallback
