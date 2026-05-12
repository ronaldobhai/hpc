// Build: g++ ass2_sorts.cpp -o ass2_sorts -fopenmp
// Run: ./ass2_sorts

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

/* ================= CHECK SORTED ================= */

bool isSorted(vector<int>& arr)
{
    for(int i = 1; i < arr.size(); i++)
        if(arr[i] < arr[i - 1])
            return false;

    return true;
}

/* ================= BUBBLE SORT ================= */

void sequentialBubbleSort(vector<int>& arr)
{
    int n = arr.size();

    for(int i = 0; i < n - 1; i++)
    {
        for(int j = 0; j < n - i - 1; j++)
        {
            if(arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

void parallelBubbleSort(vector<int>& arr)
{
    int n = arr.size();

    for(int i = 0; i < n; i++)
    {
        if(i % 2 == 0)
        {
            #pragma omp parallel for
            for(int j = 0; j < n - 1; j += 2)
            {
                if(arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
        else
        {
            #pragma omp parallel for
            for(int j = 1; j < n - 1; j += 2)
            {
                if(arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
    }
}

/* ================= MERGE FUNCTION ================= */

void merge(vector<int>& arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for(int i = 0; i < n1; i++)
        L[i] = arr[l + i];

    for(int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while(i < n1 && j < n2)
    {
        if(L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while(i < n1)
        arr[k++] = L[i++];

    while(j < n2)
        arr[k++] = R[j++];
}

/* ================= MERGE SORT ================= */

void sequentialMergeSort(vector<int>& arr, int l, int r)
{
    if(l < r)
    {
        int m = l + (r - l) / 2;

        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int>& arr, int l, int r)
{
    if(l >= r)
        return;

    // Small subarrays -> sequential
    if(r - l < 1000)
    {
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

/* ================= MAIN ================= */

int main()
{
    // Existing dataset size
    int n = 50000;

    vector<int> arr(n);

    // Generate random dataset
    srand(time(0));

    for(int i = 0; i < n; i++)
        arr[i] = rand() % 100000;

    // Copies for different algorithms
    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    int threads = omp_get_max_threads();

    cout << "===== OPENMP SORT PERFORMANCE =====\n";
    cout << "Dataset Size : " << n << endl;
    cout << "Threads Used : " << threads << endl;

    double start, end;

    /* ================= SEQUENTIAL BUBBLE ================= */

    start = omp_get_wtime();

    sequentialBubbleSort(arr1);

    end = omp_get_wtime();

    double seqBubble = end - start;

    /* ================= PARALLEL BUBBLE ================= */

    start = omp_get_wtime();

    parallelBubbleSort(arr2);

    end = omp_get_wtime();

    double parBubble = end - start;

    /* ================= SEQUENTIAL MERGE ================= */

    start = omp_get_wtime();

    sequentialMergeSort(arr3, 0, n - 1);

    end = omp_get_wtime();

    double seqMerge = end - start;

    /* ================= PARALLEL MERGE ================= */

    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(arr4, 0, n - 1);
    }

    end = omp_get_wtime();

    double parMerge = end - start;

    /* ================= RESULTS ================= */

    cout << "\n===== EXECUTION TIMES =====\n";

    cout << "\nSequential Bubble Sort : "
         << seqBubble << " sec";

    cout << "\nParallel Bubble Sort   : "
         << parBubble << " sec";

    cout << "\n\nSequential Merge Sort  : "
         << seqMerge << " sec";

    cout << "\nParallel Merge Sort    : "
         << parMerge << " sec";

    /* ================= SPEEDUP ================= */

    double bubbleSpeedup = seqBubble / parBubble;
    double mergeSpeedup  = seqMerge / parMerge;

    cout << "\n\n===== SPEEDUP =====\n";

    cout << "\nBubble Sort Speedup : "
         << bubbleSpeedup;

    cout << "\nMerge Sort Speedup  : "
         << mergeSpeedup;

    /* ================= EFFICIENCY ================= */

    cout << "\n\n===== EFFICIENCY =====\n";

    cout << "\nBubble Efficiency : "
         << bubbleSpeedup / threads;

    cout << "\nMerge Efficiency  : "
         << mergeSpeedup / threads;

    /* ================= THROUGHPUT ================= */

    cout << "\n\n===== THROUGHPUT =====\n";

    cout << "\nBubble Throughput : "
         << n / parBubble
         << " elements/sec";

    cout << "\nMerge Throughput  : "
         << n / parMerge
         << " elements/sec";

    /* ================= SORT VALIDATION ================= */

    cout << "\n\n===== SORT CHECK =====\n";

    cout << "\nSequential Bubble : "
         << isSorted(arr1);

    cout << "\nParallel Bubble   : "
         << isSorted(arr2);

    cout << "\nSequential Merge  : "
         << isSorted(arr3);

    cout << "\nParallel Merge    : "
         << isSorted(arr4);

    /* ================= DISPLAY SAMPLE OUTPUT ================= */

    cout << "\n\n===== FIRST 20 SORTED ELEMENTS =====\n";

    for(int i = 0; i < min(20, n); i++)
        cout << arr4[i] << " ";

    cout << endl;

    return 0;
}