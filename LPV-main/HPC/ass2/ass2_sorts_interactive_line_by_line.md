# ass2_sorts_interactive.cpp - Line by Line Explanation

This version of the sorting program asks the user for the array size and then sorts random data.

## 1. Build and run comments
- Line 1 shows the compile command with OpenMP.
- Line 2 shows how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard C++ headers.
- `#include <iostream>` is used for input and output.
- `#include <vector>` is used for storing the array.
- `#include <cstdlib>` is used for random numbers.
- `#include <ctime>` is used to seed random numbers.
- `#include <omp.h>` is used for OpenMP timing and parallel code.

## 3. Example comments
- The example input comment shows how to test the program.
- The expected output comment shows what a correct result looks like.
- These comments help users verify the code without reading all the logic.

## 4. Namespace
- `using namespace std;` keeps the code shorter and easier to read.

## 5. isSorted function
- This function checks whether the array is in ascending order.
- It loops through the array from the second element onward.
- If any value is smaller than the previous one, the function returns `false`.
- If the loop finishes, the array is sorted and the function returns `true`.
- This is a simple correctness check.

## 6. sequentialBubbleSort
- This function sorts the array using the classic Bubble Sort method.
- It repeatedly compares neighboring values.
- If the left value is larger, it swaps them.
- After each pass, the largest remaining value moves to the end.
- The algorithm is easy to understand but slow for large data.

## 7. parallelBubbleSort
- This function uses OpenMP to parallelize the pair comparisons.
- The outer loop repeats many passes.
- On even passes, the code checks pairs starting from index 0.
- On odd passes, the code checks pairs starting from index 1.
- `#pragma omp parallel for` lets several threads work on different pairs at the same time.
- This is an odd-even style parallel bubble sort.
- It still has a lot of overhead, so it may be slower for small arrays.

## 8. merge function
- `merge()` combines two already sorted halves.
- `L` stores the left half.
- `R` stores the right half.
- The code copies values into `L` and `R` first.
- Then it compares the current values from both halves and writes the smaller one into the original array.
- The remaining values from either side are copied at the end.
- This is the important merge step used by Merge Sort.

## 9. sequentialMergeSort
- This is the normal recursive Merge Sort.
- It stops when the part of the array has one element or less.
- It finds the middle index.
- It sorts the left half.
- It sorts the right half.
- It merges both halves back together.
- This divide-and-conquer method is much faster than Bubble Sort.

## 10. parallelMergeSort
- This version uses OpenMP tasks.
- It first checks whether the part is too small.
- If the part is small, it uses sequential Merge Sort because parallel overhead would be too high.
- For larger parts, it splits the array into two halves.
- `#pragma omp task shared(arr)` creates one task for each half.
- `#pragma omp taskwait` waits for both tasks before merging.
- This allows the left and right halves to sort at the same time.

## 11. main function input
- `int n;` stores the array size from the user.
- The program asks the user to enter a size.
- `cin >> n;` reads the value.
- `if (n <= 0)` checks that the size is valid.
- If the size is invalid, the program prints an error and exits.
- This prevents memory or logic errors.

## 12. Array creation and random data
- `vector<int> arr(n);` creates an array of size `n`.
- `srand(time(0));` seeds the random number generator.
- The program prints that it is generating data.
- The loop fills the array with random integers.
- The data range is 0 to 99,999.
- If the array is small enough, the program prints the values before sorting.

## 13. Copies for testing
- `arr1`, `arr2`, `arr3`, and `arr4` are copies of the same input array.
- Each sorting method gets identical data.
- This makes the performance comparison fair.

## 14. Thread count
- `omp_get_max_threads()` returns how many threads OpenMP can use.
- This value is printed to the screen.
- It is used later when efficiency is calculated.

## 15. Timing Bubble Sort
- The program measures sequential bubble sort first.
- Then it measures parallel bubble sort.
- `omp_get_wtime()` is used to measure elapsed time.
- The times are saved as `seqBubble` and `parBubble`.
- These values are later used for speedup.

## 16. Timing Merge Sort
- The program measures sequential Merge Sort on one copy.
- Then it measures parallel Merge Sort.
- The parallel Merge Sort starts in a parallel region.
- `#pragma omp single` makes one thread start the recursive task tree.
- The recursive calls themselves create more tasks.
- This is a common way to use OpenMP tasks.

## 17. Printing results
- The program prints the execution times for all four sorts.
- It then computes the speedups.
- It also computes efficiency for both bubble and merge sort.
- Throughput is shown as elements per second.

## 18. Validation
- The program checks each sorted array with `isSorted()`.
- The printed values should be `1` if sorting worked.
- This is an easy way to confirm correctness.

## 19. Sorted array display
- If `n <= 20`, the program prints the final sorted array.
- This is useful for small tests.
- It helps the user visually check the output.

## 20. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This file is the user-input version of the sorting demo.
- It sorts random numbers and compares sequential and parallel methods.
- Merge Sort benefits much more from OpenMP than Bubble Sort.