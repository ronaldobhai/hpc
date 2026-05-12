// Build: g++ ass3_parallel_reduction.cpp -o ass3_parallel_reduction -fopenmp
// Run: ./ass3_parallel_reduction

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

/* ================= SEQUENTIAL OPERATIONS ================= */

int sequentialSum(vector<int>& arr)
{
    int sum = 0;

    for(int i = 0; i < arr.size(); i++)
        sum += arr[i];

    return sum;
}

int sequentialMin(vector<int>& arr)
{
    int min_val = arr[0];

    for(int i = 1; i < arr.size(); i++)
    {
        if(arr[i] < min_val)
            min_val = arr[i];
    }

    return min_val;
}

int sequentialMax(vector<int>& arr)
{
    int max_val = arr[0];

    for(int i = 1; i < arr.size(); i++)
    {
        if(arr[i] > max_val)
            max_val = arr[i];
    }

    return max_val;
}

double sequentialAvg(vector<int>& arr)
{
    int sum = sequentialSum(arr);

    return (double)sum / arr.size();
}

/* ================= PARALLEL OPERATIONS ================= */

int parallelSum(vector<int>& arr)
{
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < arr.size(); i++)
        sum += arr[i];

    return sum;
}

int parallelMin(vector<int>& arr)
{
    int min_val = arr[0];

    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < arr.size(); i++)
    {
        if(arr[i] < min_val)
            min_val = arr[i];
    }

    return min_val;
}

int parallelMax(vector<int>& arr)
{
    int max_val = arr[0];

    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < arr.size(); i++)
    {
        if(arr[i] > max_val)
            max_val = arr[i];
    }

    return max_val;
}

double parallelAvg(vector<int>& arr)
{
    int sum = parallelSum(arr);

    return (double)sum / arr.size();
}

/* ================= MAIN ================= */

int main()
{
    // Existing dataset size
    int n = 1000000;

    vector<int> arr(n);

    // Generate random dataset
    srand(time(0));

    for(int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    int threads = omp_get_max_threads();

    cout << "===== OPENMP PARALLEL REDUCTION =====\n";

    cout << "\nDataset Size : " << n;
    cout << "\nThreads Used : " << threads;

    double start, end;

    /* ================= SUM ================= */

    start = omp_get_wtime();

    int seqSum = sequentialSum(arr);

    end = omp_get_wtime();

    double seqSumTime = end - start;

    start = omp_get_wtime();

    int parSum = parallelSum(arr);

    end = omp_get_wtime();

    double parSumTime = end - start;

    /* ================= MIN ================= */

    start = omp_get_wtime();

    int seqMin = sequentialMin(arr);

    end = omp_get_wtime();

    double seqMinTime = end - start;

    start = omp_get_wtime();

    int parMin = parallelMin(arr);

    end = omp_get_wtime();

    double parMinTime = end - start;

    /* ================= MAX ================= */

    start = omp_get_wtime();

    int seqMax = sequentialMax(arr);

    end = omp_get_wtime();

    double seqMaxTime = end - start;

    start = omp_get_wtime();

    int parMax = parallelMax(arr);

    end = omp_get_wtime();

    double parMaxTime = end - start;

    /* ================= AVG ================= */

    start = omp_get_wtime();

    double seqAvg = sequentialAvg(arr);

    end = omp_get_wtime();

    double seqAvgTime = end - start;

    start = omp_get_wtime();

    double parAvg = parallelAvg(arr);

    end = omp_get_wtime();

    double parAvgTime = end - start;

    /* ================= RESULTS ================= */

    cout << "\n\n===== FINAL RESULTS =====\n";

    cout << "\nSum      : " << parSum;
    cout << "\nMinimum  : " << parMin;
    cout << "\nMaximum  : " << parMax;
    cout << "\nAverage  : " << parAvg;

    /* ================= EXECUTION TIME ================= */

    cout << "\n\n===== EXECUTION TIMES =====\n";

    cout << "\nSequential Sum Time : "
         << seqSumTime << " sec";

    cout << "\nParallel Sum Time   : "
         << parSumTime << " sec";

    cout << "\n\nSequential Min Time : "
         << seqMinTime << " sec";

    cout << "\nParallel Min Time   : "
         << parMinTime << " sec";

    cout << "\n\nSequential Max Time : "
         << seqMaxTime << " sec";

    cout << "\nParallel Max Time   : "
         << parMaxTime << " sec";

    cout << "\n\nSequential Avg Time : "
         << seqAvgTime << " sec";

    cout << "\nParallel Avg Time   : "
         << parAvgTime << " sec";

    /* ================= PERFORMANCE METRICS ================= */

    double speedup = seqSumTime / parSumTime;

    double efficiency = speedup / threads;

    double throughput = n / parSumTime;

    cout << "\n\n===== PERFORMANCE METRICS =====\n";

    cout << "\nSpeedup    : "
         << speedup;

    cout << "\nEfficiency : "
         << efficiency;

    cout << "\nThroughput : "
         << throughput
         << " elements/sec";

    return 0;
}