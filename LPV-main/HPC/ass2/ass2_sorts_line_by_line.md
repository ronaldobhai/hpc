# ass2_sorts.cpp - Line by Line Explanation

This file compares sequential and parallel Bubble Sort and Merge Sort on a large random array.

## 1. Build and run comments
- Line 1 shows the compile command using `g++` and OpenMP.
- Line 2 shows the run command.

## 2. Library includes
- `#include <bits/stdc++.h>` pulls in most standard C++ headers.
- `#include <iostream>` is used for printing results.
- `#include <vector>` is used to store the array data.
- `#include <cstdlib>` is used for `rand()` and `srand()`.
- `#include <ctime>` is used for `time(0)`.
- `#include <omp.h>` is used for OpenMP timing and parallel loops.

## 3. Namespace
- `using namespace std;` lets the program use standard names without writing `std::` all the time.

## 4. isSorted function
- `bool isSorted(vector<int>& arr)` checks whether the array is sorted in ascending order.
- The loop starts from index 1 because each element is compared with the one before it.
- If any element is smaller than the previous one, the array is not sorted.
- The function returns `false` in that case.
- If the loop finishes without finding a problem, it returns `true`.
- This is used to validate the sorting result.

## 5. Sequential Bubble Sort
- `void sequentialBubbleSort(vector<int>& arr)` sorts the array using the normal Bubble Sort method.
- `int n = arr.size();` stores the array length.
- The outer loop controls how many passes are made.
- The inner loop compares neighboring values.
- If the left value is bigger than the right value, `swap()` exchanges them.
- After every pass, the biggest unsorted value moves to the right side.
- This is simple but slow for large arrays.

## 6. Parallel Bubble Sort
- `void parallelBubbleSort(vector<int>& arr)` tries to parallelize Bubble Sort using odd-even style passes.
- The outer loop repeats for `n` passes.
- On even passes, the code compares pairs starting at index 0.
- On odd passes, it compares pairs starting at index 1.
- `#pragma omp parallel for` splits the pair comparisons across threads.
- This works better than naive Bubble Sort parallelization because each thread handles independent pairs in one pass.
- However, bubble sort is still expensive overall.
- For small arrays, OpenMP overhead can make it slower than the sequential version.

## 7. merge function
- `void merge(vector<int>& arr, int l, int m, int r)` combines two sorted halves into one sorted section.
- `n1` is the size of the left half.
- `n2` is the size of the right half.
- `L` and `R` are temporary arrays for those halves.
- The first `for` loop copies the left half into `L`.
- The second `for` loop copies the right half into `R`.
- `i`, `j`, and `k` are indexes for the left array, right array, and original array.
- The `while` loop compares one value from each half and writes the smaller one back to `arr`.
- The next two loops copy any remaining items after one half is finished.
- This is the standard merge step used in Merge Sort.

## 8. Sequential Merge Sort
- `void sequentialMergeSort(vector<int>& arr, int l, int r)` sorts a portion of the array recursively.
- `if (l < r)` checks that there is more than one item.
- `int m = l + (r - l) / 2;` finds the middle index safely.
- The function sorts the left half first.
- Then it sorts the right half.
- Finally it merges the two sorted halves.
- This is the usual recursive divide-and-conquer Merge Sort.

## 9. Parallel Merge Sort
- `void parallelMergeSort(vector<int>& arr, int l, int r)` is the parallel version.
- `if (l >= r)` stops when the section has zero or one item.
- `if (r - l < 1000)` switches back to sequential Merge Sort for small sections.
- That small-size cutoff avoids too much OpenMP overhead.
- The middle point is computed the same way as before.
- `#pragma omp task shared(arr)` creates a parallel task for the left half.
- Another task is created for the right half.
- `#pragma omp taskwait` waits for both tasks to finish before merging.
- This is a good use of task parallelism because the two halves are independent.

## 10. main function setup
- `int n = 50000;` sets the array size to 50,000.
- `vector<int> arr(n);` creates the array.
- `srand(time(0));` seeds the random number generator.
- The `for` loop fills the array with random values from 0 to 99,999.
- `arr1`, `arr2`, `arr3`, and `arr4` are copies so each algorithm sorts the same data.
- `int threads = omp_get_max_threads();` stores the available OpenMP thread count.

## 11. Printing the header
- The program prints a title showing it is a performance test.
- It prints the dataset size and thread count.
- This helps compare results across runs.

## 12. Timing sequential and parallel bubble sort
- `start = omp_get_wtime();` starts the clock.
- `sequentialBubbleSort(arr1);` sorts the first copy.
- `end = omp_get_wtime();` stops the clock.
- `seqBubble` stores the time difference.
- The same process is repeated for `parallelBubbleSort(arr2);`.
- The times can be very different because Bubble Sort is expensive and parallel overhead is large.

## 13. Timing sequential and parallel merge sort
- The sequential merge sort is timed the same way on `arr3`.
- For parallel merge sort, the code opens a parallel region first.
- `#pragma omp single` makes only one thread start the initial recursive call.
- The recursive function then creates tasks for other threads.
- The result is stored in `arr4`.

## 14. Printing execution times
- The program prints the sequential and parallel time for Bubble Sort.
- It also prints the sequential and parallel time for Merge Sort.
- This lets the user compare both algorithms.

## 15. Speedup calculations
- `bubbleSpeedup = seqBubble / parBubble;` measures how much faster parallel bubble sort was.
- `mergeSpeedup = seqMerge / parMerge;` measures merge sort speedup.
- A value above 1 means the parallel version was faster.
- A value below 1 means it was slower.

## 16. Efficiency calculations
- Efficiency divides speedup by the number of threads.
- This shows how well the threads were used.
- Perfect efficiency would be 1.0, which is rare in real programs.
- Bubble sort usually has low efficiency because of overhead.

## 17. Throughput calculations
- Throughput means how many elements were sorted per second.
- `n / parBubble` gives bubble sort throughput.
- `n / parMerge` gives merge sort throughput.
- Bigger throughput means better performance.

## 18. Validation section
- The program prints whether each result is sorted.
- `isSorted(arr1)` checks sequential bubble sort.
- `isSorted(arr2)` checks parallel bubble sort.
- `isSorted(arr3)` checks sequential merge sort.
- `isSorted(arr4)` checks parallel merge sort.
- The result should be `1` for all four if the sorts worked.

## 19. Printing the sorted array
- If the array size is small, the program prints the final sorted values.
- This is only for easy checking when the input is small.
- For a 50,000-element array, printing the whole array would be too much.

## 20. Final return
- `return 0;` ends the program successfully.

## Simple summary
- Bubble Sort is simple but slow.
- Merge Sort is much faster.
- OpenMP tasks help Merge Sort more than Bubble Sort.
- The validation step confirms the output is really sorted.