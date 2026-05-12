// Build: g++ ass2_sorts_interactive.cpp -o ass2_sorts_interactive -fopenmp
// Run: ./ass2_sorts_interactive

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// Example Input:
// Enter array size: 10
// Generated array: 45 23 67 12 89 34 56 78 90 11

// Expected Output:
// Sequential Bubble: 1 1
// Parallel Bubble: 1 1
// Sequential Merge: 1 1
// Parallel Merge: 1 1
// Sorted array: 11 12 23 34 45 56 67 78 89 90

using namespace std;

bool isSorted(vector<int>& arr) {
    for (int i = 1; i < arr.size(); i++)
        if (arr[i] < arr[i - 1])
            return false;
    return true;
}

void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
#pragma omp parallel for
            for (int j = 0; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        } else {
#pragma omp parallel for
            for (int j = 1; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

void sequentialMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l >= r)
        return;

    if (r - l < 1000) {
        sequentialMergeSort(arr, l, r);
        return;
    }

    int m = l + (r - l) / 2;

#pragma omp task shared(arr)
    parallelMergeSort(arr, l, m);

#pragma omp task shared(arr)
    parallelMergeSort(arr, m + 1, r);

#pragma omp taskwait

    merge(arr, l, m, r);
}

int main() {
    int n;
    cout << "Enter array size: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid size!\n";
        return 1;
    }

    vector<int> arr(n);
    srand(time(0));

    cout << "\nGenerating random array...\n";
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 100000;

    if (n <= 20) {
        cout << "Array: ";
        for (int x : arr)
            cout << x << " ";
        cout << "\n";
    }

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    int threads = omp_get_max_threads();

    cout << "\n===== OPENMP SORT PERFORMANCE =====\n";
    cout << "Dataset Size : " << n << endl;
    cout << "Threads Used : " << threads << endl;

    double start, end;

    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();
    double seqBubble = end - start;

    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    double parBubble = end - start;

    start = omp_get_wtime();
    sequentialMergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();
    double seqMerge = end - start;

    start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        parallelMergeSort(arr4, 0, n - 1);
    }
    end = omp_get_wtime();
    double parMerge = end - start;

    cout << "\n===== EXECUTION TIMES =====\n";
    cout << "\nSequential Bubble Sort : " << seqBubble << " sec";
    cout << "\nParallel Bubble Sort   : " << parBubble << " sec";
    cout << "\n\nSequential Merge Sort  : " << seqMerge << " sec";
    cout << "\nParallel Merge Sort    : " << parMerge << " sec";

    double bubbleSpeedup = seqBubble / parBubble;
    double mergeSpeedup = seqMerge / parMerge;

    cout << "\n\n===== SPEEDUP =====\n";
    cout << "\nBubble Sort Speedup : " << bubbleSpeedup;
    cout << "\nMerge Sort Speedup  : " << mergeSpeedup;

    cout << "\n\n===== EFFICIENCY =====\n";
    cout << "\nBubble Efficiency : " << bubbleSpeedup / threads;
    cout << "\nMerge Efficiency  : " << mergeSpeedup / threads;

    cout << "\n\n===== SORT VALIDATION =====\n";
    cout << "\nSequential Bubble : " << isSorted(arr1);
    cout << "\nParallel Bubble   : " << isSorted(arr2);
    cout << "\nSequential Merge  : " << isSorted(arr3);
    cout << "\nParallel Merge    : " << isSorted(arr4);

    if (n <= 20) {
        cout << "\n\n===== SORTED ARRAY =====\n";
        for (int i = 0; i < n; i++)
            cout << arr4[i] << " ";
        cout << endl;
    }

    return 0;
}
