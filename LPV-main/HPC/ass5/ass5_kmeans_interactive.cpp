// Build: g++ ass5_kmeans_interactive.cpp -o ass5_kmeans_interactive -fopenmp -lm
// Run: ./ass5_kmeans_interactive

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// Example Input:
// Enter number of data points: 10
// Enter number of clusters: 2
// Enter number of iterations: 5
// Generate random data? (1=yes, 0=manual): 1

// Expected Output:
// Cluster assignments and timing for different thread counts
// Speedup and efficiency metrics

using namespace std;

struct Point {
    float x, y;
    int cluster;
};

float distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void kmeans_parallel(vector<Point>& points, vector<Point>& centroids, int k,
                     int iterations) {
    int n = points.size();

    for (int it = 0; it < iterations; it++) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float minDist = 1e9;
            int clusterIndex = 0;

            for (int j = 0; j < k; j++) {
                float dist = distance(points[i], centroids[j]);

                if (dist < minDist) {
                    minDist = dist;
                    clusterIndex = j;
                }
            }

            points[i].cluster = clusterIndex;
        }

        vector<float> sumX(k, 0);
        vector<float> sumY(k, 0);
        vector<int> count(k, 0);

        for (int i = 0; i < n; i++) {
            int c = points[i].cluster;
            sumX[c] += points[i].x;
            sumY[c] += points[i].y;
            count[c]++;
        }

        for (int i = 0; i < k; i++) {
            if (count[i] != 0) {
                centroids[i].x = sumX[i] / count[i];
                centroids[i].y = sumY[i] / count[i];
            }
        }
    }
}

int main() {
    int n, k, iterations;

    cout << "Enter number of data points: ";
    cin >> n;

    cout << "Enter number of clusters: ";
    cin >> k;

    cout << "Enter number of iterations: ";
    cin >> iterations;

    if (n <= 0 || k <= 0 || iterations <= 0 || k > n) {
        cout << "Invalid input!\n";
        return 1;
    }

    vector<Point> original(n);

    int choice;
    cout << "Generate random data? (1=yes, 0=manual): ";
    cin >> choice;

    if (choice == 1) {
        srand(time(0));
        cout << "\nGenerating random data...\n";
        for (int i = 0; i < n; i++) {
            original[i].x = rand() % 100;
            original[i].y = rand() % 100;
            original[i].cluster = 0;
        }
    } else {
        cout << "\nEnter points (x y) for each point:\n";
        for (int i = 0; i < n; i++) {
            cout << "Point " << i + 1 << ": ";
            cin >> original[i].x >> original[i].y;
            original[i].cluster = 0;
        }
    }

    if (n <= 20) {
        cout << "\nData points:\n";
        for (int i = 0; i < n; i++) {
            cout << "(" << original[i].x << ", " << original[i].y << ") ";
        }
        cout << "\n";
    }

    int thread_counts[] = {1, 2, 4};

    cout << "\n===== HPC APPLICATION FOR AI/ML =====\n";
    cout << "\nAlgorithm : Parallel K-Means Clustering";
    cout << "\nDataset Size : " << n;
    cout << "\nClusters : " << k;
    cout << "\nIterations : " << iterations;

    cout << "\n\n===== THREAD SCALING PERFORMANCE =====\n";

    for (int t : thread_counts) {
        vector<Point> points = original;
        vector<Point> centroids(k);

        for (int i = 0; i < k; i++) {
            centroids[i] = points[rand() % n];
        }

        omp_set_num_threads(t);

        cout << "\nThread count: " << t << "\n";
        cout << "Initial centroids:\n";
        for (int i = 0; i < k; i++) {
            cout << "  C" << i << ": (" << centroids[i].x << ", " << centroids[i].y
                 << ")\n";
        }

        double start = omp_get_wtime();
        kmeans_parallel(points, centroids, k, iterations);
        double end = omp_get_wtime();

        double executionTime = end - start;
        double throughput = n / executionTime;

        cout << "Execution Time : " << executionTime << " seconds";
        cout << "\nThroughput : " << throughput << " points/sec";

        cout << "\nFinal centroids:\n";
        for (int i = 0; i < k; i++) {
            cout << "  C" << i << ": (" << centroids[i].x << ", " << centroids[i].y
                 << ")\n";
        }

        if (n <= 20) {
            cout << "Final cluster assignments: ";
            for (int i = 0; i < n; i++) {
                cout << points[i].cluster << " ";
            }
            cout << "\n";
        }

        cout << "\n";
    }

    return 0;
}
