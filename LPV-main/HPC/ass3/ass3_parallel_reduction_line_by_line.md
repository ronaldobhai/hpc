# ass3_parallel_reduction.cpp - Line by Line Explanation

This file compares sequential and parallel versions of sum, minimum, maximum, and average on a large random array.

## 1. Build and run comments
- Line 1 tells you how to compile with OpenMP.
- Line 2 tells you how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard C++ headers.
- `#include <iostream>` is used for printing.
- `#include <vector>` stores the number list.
- `#include <cstdlib>` gives access to `rand()` and `srand()`.
- `#include <ctime>` gives access to `time(0)`.
- `#include <omp.h>` is used for OpenMP timing and reduction.

## 3. Namespace
- `using namespace std;` lets the file use standard names without `std::`.

## 4. Sequential sum
- `int sequentialSum(vector<int>& arr)` adds all values one by one.
- `int sum = 0;` stores the running total.
- The `for` loop walks through the entire array.
- Each value is added to `sum`.
- At the end, the total is returned.
- This is the baseline single-thread version.

## 5. Sequential minimum
- `int sequentialMin(vector<int>& arr)` finds the smallest value.
- `int min_val = arr[0];` starts with the first element.
- The loop begins at index 1 because the first element is already the starting minimum.
- If a smaller number is found, `min_val` is updated.
- The final minimum is returned.

## 6. Sequential maximum
- `int sequentialMax(vector<int>& arr)` finds the largest value.
- `int max_val = arr[0];` starts with the first element.
- The loop checks every other element.
- If a bigger value is found, it replaces `max_val`.
- The final maximum is returned.

## 7. Sequential average
- `double sequentialAvg(vector<int>& arr)` computes the average.
- It calls `sequentialSum(arr)` to get the total.
- The total is divided by `arr.size()`.
- The result is cast to `double` so decimals are kept.
- This is the basic sequential average.

## 8. Parallel sum
- `int parallelSum(vector<int>& arr)` uses OpenMP reduction.
- `int sum = 0;` starts the shared result.
- `#pragma omp parallel for reduction(+:sum)` tells OpenMP to split the loop across threads.
- Each thread gets a private copy of `sum` during the loop.
- At the end, OpenMP combines the partial sums into one final result.
- This avoids race conditions.

## 9. Parallel minimum
- `int parallelMin(vector<int>& arr)` also uses reduction.
- `int min_val = arr[0];` sets the starting value.
- `reduction(min:min_val)` tells OpenMP to keep the smallest value from all threads.
- Each thread compares values inside its own part of the array.
- OpenMP combines the results safely at the end.
- This is better than manually locking every comparison.

## 10. Parallel maximum
- `int parallelMax(vector<int>& arr)` works like parallel minimum.
- `int max_val = arr[0];` is the starting value.
- `reduction(max:max_val)` tells OpenMP to keep the largest value from all threads.
- Each thread scans a part of the array.
- OpenMP merges the results into one final maximum.

## 11. Parallel average
- `double parallelAvg(vector<int>& arr)` computes average using the parallel sum.
- It calls `parallelSum(arr)` first.
- Then it divides by the array size.
- The average itself is not parallelized separately because the sum is the expensive part.
- This gives the same result as the sequential average.

## 12. main function setup
- `int n = 1000000;` makes a large array with one million values.
- `vector<int> arr(n);` creates the array storage.
- `srand(time(0));` seeds the random generator.
- The loop fills the array with random numbers from 0 to 9,999.
- `int threads = omp_get_max_threads();` stores the number of available OpenMP threads.

## 13. Printing the header
- The program prints a title showing that it is testing parallel reduction.
- It prints the dataset size.
- It prints the number of threads being used.
- This gives context before the timings appear.

## 14. Timing sum
- `omp_get_wtime()` starts and stops the timer around each sum version.
- `seqSum` stores the sequential result.
- `parSum` stores the parallel result.
- `seqSumTime` and `parSumTime` store the elapsed time for each version.
- This lets the user compare performance.

## 15. Timing min, max, and average
- The same timing pattern is repeated for minimum, maximum, and average.
- Each operation is measured in sequential and parallel form.
- The code keeps the times in separate variables.
- This makes the final report easy to read.

## 16. Printing final results
- The final output prints the parallel sum, minimum, maximum, and average.
- The parallel answers should match the sequential answers.
- This confirms correctness.

## 17. Printing execution times
- The program prints each sequential time and each parallel time.
- This includes sum, min, max, and average.
- The user can see which operations benefit most from parallelism.

## 18. Performance metrics
- `speedup = seqSumTime / parSumTime` shows how much faster the sum became.
- `efficiency = speedup / threads` shows how well the threads were used.
- `throughput = n / parSumTime` shows how many numbers were processed per second.
- These metrics are useful for understanding parallel performance.

## 19. Final return
- `return 0;` ends the program.

## Simple summary
- This file teaches OpenMP reduction.
- Reduction is the cleanest way to do safe parallel sum, min, and max.
- It is much better than using shared variables without protection.