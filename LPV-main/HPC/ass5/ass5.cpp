// Build: g++ ass5.cpp -o ass5 -fopenmp
// Run: ./ass5

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

/* ================= POINT STRUCTURE ================= */

struct Point
{
    float x, y;
    int cluster;
};

/* ================= DISTANCE FUNCTION ================= */

float distance(Point a, Point b)
{
    return sqrt(
        (a.x - b.x) * (a.x - b.x) +
        (a.y - b.y) * (a.y - b.y)
    );
}

/* ================= PARALLEL K-MEANS ================= */

void kmeans_parallel(
    vector<Point>& points,
    vector<Point>& centroids,
    int k,
    int iterations)
{
    int n = points.size();

    for(int it = 0; it < iterations; it++)
    {
        /* ================= CLUSTER ASSIGNMENT ================= */

        #pragma omp parallel for
        for(int i = 0; i < n; i++)
        {
            float minDist = 1e9;

            int clusterIndex = 0;

            for(int j = 0; j < k; j++)
            {
                float dist = distance(points[i], centroids[j]);

                if(dist < minDist)
                {
                    minDist = dist;
                    clusterIndex = j;
                }
            }

            points[i].cluster = clusterIndex;
        }

        /* ================= UPDATE CENTROIDS ================= */

        vector<float> sumX(k, 0);
        vector<float> sumY(k, 0);

        vector<int> count(k, 0);

        for(int i = 0; i < n; i++)
        {
            int c = points[i].cluster;

            sumX[c] += points[i].x;
            sumY[c] += points[i].y;

            count[c]++;
        }

        for(int i = 0; i < k; i++)
        {
            if(count[i] != 0)
            {
                centroids[i].x = sumX[i] / count[i];

                centroids[i].y = sumY[i] / count[i];
            }
        }
    }
}

/* ================= MAIN ================= */

int main()
{
    // Number of data points
    int n = 100000;

    // Number of clusters
    int k = 3;

    // Number of iterations
    int iterations = 10;

    vector<Point> original(n);

    srand(time(0));

    /* ================= GENERATE RANDOM DATASET ================= */

    for(int i = 0; i < n; i++)
    {
        original[i].x = rand() % 100;

        original[i].y = rand() % 100;

        original[i].cluster = 0;
    }

    /* ================= THREAD SCALING ================= */

    int thread_counts[] = {1, 2, 4, 8};

    cout << "===== HPC APPLICATION FOR AI/ML =====\n";

    cout << "\nAlgorithm : Parallel K-Means Clustering";
    cout << "\nDataset Size : " << n;
    cout << "\nClusters : " << k;
    cout << "\nIterations : " << iterations;

    cout << "\n\n===== THREAD SCALING PERFORMANCE =====\n";

    for(int t : thread_counts)
    {
        // Copy original dataset
        vector<Point> points = original;

        vector<Point> centroids(k);

        /* ================= INITIALIZE CENTROIDS ================= */

        for(int i = 0; i < k; i++)
        {
            centroids[i] = points[rand() % n];
        }

        // Set OpenMP thread count
        omp_set_num_threads(t);

        double start = omp_get_wtime();

        // Run K-Means
        kmeans_parallel(points, centroids, k, iterations);

        double end = omp_get_wtime();

        double executionTime = end - start;

        double throughput = n / executionTime;

        cout << "\nThreads Used : " << t;
        cout << "\nExecution Time : "
             << executionTime
             << " seconds";

        cout << "\nThroughput : "
             << throughput
             << " points/sec\n";
    }

    return 0;
}