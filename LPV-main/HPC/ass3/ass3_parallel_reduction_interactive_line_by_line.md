# ass3_parallel_reduction_interactive.cpp - Line by Line Explanation

This version asks the user for the array size and then performs sum, min, max, and average on random data.

## 1. Build and run comments
- Line 1 shows the compile command with `g++` and OpenMP.
- Line 2 shows how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard headers.
- `#include <iostream>` is for input and output.
- `#include <vector>` stores the numbers.
- `#include <cstdlib>` is for `rand()` and `srand()`.
- `#include <ctime>` is for `time(0)`.
- `#include <omp.h>` is for timing and reductions.

## 3. Example comments
- The example input shows how to test the program.
- The expected output shows the kind of answer to expect.
- This helps users learn and verify the code.

## 4. Namespace
- `using namespace std;` makes the code shorter and easier to read.

## 5. sequentialSum
- This function adds all array values one by one.
- `sum` starts at zero.
- The loop reads every element.
- Each value is added to the total.
- The final total is returned.

## 6. sequentialMin
- This function finds the smallest number in the array.
- It starts with the first value as the current minimum.
- The loop checks the rest of the values.
- If a smaller value is found, it replaces the minimum.
- The final minimum is returned.

## 7. sequentialMax
- This function finds the biggest number.
- It starts with the first array value.
- The loop checks the rest of the values.
- If a bigger number is found, it becomes the new maximum.
- The final maximum is returned.

## 8. sequentialAvg
- This function computes the average.
- It first calls `sequentialSum()`.
- Then it divides the sum by the number of items.
- The result is returned as a `double` so decimal values are kept.

## 9. parallelSum
- `#pragma omp parallel for reduction(+:sum)` splits the sum loop across threads.
- Each thread gets a private partial sum.
- OpenMP adds the partial sums together safely.
- This avoids manual locking and race conditions.

## 10. parallelMin
- `reduction(min:min_val)` tells OpenMP to keep the smallest value.
- Each thread checks a different portion of the array.
- OpenMP combines the thread results at the end.
- This is the safe way to parallelize minimum.

## 11. parallelMax
- `reduction(max:max_val)` keeps the largest value from all threads.
- The array is split among the threads.
- OpenMP combines the results after the loop.
- This gives the correct maximum.

## 12. parallelAvg
- This function uses `parallelSum()` and then divides by the array size.
- It does not need a separate parallel loop because the sum is the main work.
- The average is returned as a decimal value.

## 13. main function input
- `int n;` stores the array size typed by the user.
- The program asks for the size.
- `cin >> n;` reads it.
- If the user enters zero or a negative number, the program stops with an error.
- This prevents bad memory allocation and divide-by-zero problems.

## 14. Array creation and random data
- `vector<int> arr(n);` makes the array.
- `srand(time(0));` starts the random number generator.
- The program prints that it is generating random values.
- The loop fills the array with numbers from 0 to 9,999.
- If the array is small, it also prints the values so the user can see them.

## 15. Thread count
- `omp_get_max_threads()` tells how many OpenMP threads can be used.
- The value is printed in the output.
- It is also used later to calculate efficiency.

## 16. Timing each operation
- The program measures sequential and parallel sum separately.
- Then it measures min, max, and average the same way.
- `omp_get_wtime()` gives accurate wall-clock timing.
- Each operation gets its own timing variables.

## 17. Final results section
- The program prints the parallel sum, minimum, maximum, and average.
- These should match the sequential answers.
- This shows the reduction code works correctly.

## 18. Execution time section
- The program prints the sequential and parallel times for all operations.
- This lets the user compare performance.
- For small arrays, the parallel version may be slower because of thread overhead.

## 19. Performance metrics section
- The program computes speedup from the sum timings.
- It computes efficiency by dividing speedup by thread count.
- It computes throughput as elements processed per second.
- These numbers help explain how effective parallel processing was.

## 20. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This is the input-based reduction demo.
- It shows that OpenMP reduction makes sum, min, and max easy and safe.
- The program is also a good example of basic performance measurement.