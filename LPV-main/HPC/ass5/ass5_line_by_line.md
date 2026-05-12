# ass5.cpp - Line by Line Explanation

This program runs parallel K-Means clustering on a large set of random 2D points and measures how the program behaves with different thread counts.

## 1. Build and run comments
- Line 1 tells you how to compile with `g++`, OpenMP, and the math library.
- Line 2 tells you how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard C++ headers.
- `#include <iostream>` is for output.
- `#include <vector>` stores points and centroids.
- `#include <cmath>` provides `sqrt()`.
- `#include <cstdlib>` provides `rand()` and `srand()`.
- `#include <ctime>` provides `time(0)`.
- `#include <omp.h>` is used for OpenMP and timing.

## 3. Namespace
- `using namespace std;` avoids writing `std::` all the time.

## 4. Point structure
- `struct Point` stores one 2D point.
- `float x, y;` are the coordinates.
- `int cluster;` stores which cluster the point belongs to.
- This structure keeps the code easy to read.

## 5. distance function
- `float distance(Point a, Point b)` computes the distance between two points.
- The formula uses the square root of the squared differences.
- This is the normal Euclidean distance formula.
- The smaller the distance, the closer the point is to a centroid.

## 6. kmeans_parallel function header
- `void kmeans_parallel(...)` does the main K-Means work.
- It receives the points, centroids, cluster count `k`, and number of iterations.
- `int n = points.size();` stores how many points there are.

## 7. Outer iteration loop
- `for (int it = 0; it < iterations; it++)` repeats K-Means steps.
- Each iteration has two major parts: assign points, then update centroids.
- Repeating this helps centroids move to better positions.

## 8. Cluster assignment loop
- `#pragma omp parallel for` parallelizes the loop over points.
- Each thread handles some points independently.
- `float minDist = 1e9;` starts with a very large distance.
- `int clusterIndex = 0;` stores the nearest centroid index.
- The inner loop compares the current point to every centroid.
- If a smaller distance is found, the best cluster index is updated.
- `points[i].cluster = clusterIndex;` stores the chosen cluster.
- This part is a good use of OpenMP because each point can be processed separately.

## 9. Centroid update arrays
- `vector<float> sumX(k, 0);` stores x-coordinate sums for each cluster.
- `vector<float> sumY(k, 0);` stores y-coordinate sums.
- `vector<int> count(k, 0);` stores how many points are in each cluster.
- These arrays help compute new centroid positions.

## 10. Centroid update loop
- The loop scans all points.
- `int c = points[i].cluster;` gets the cluster of the current point.
- The point's x and y values are added to that cluster's sums.
- The count for that cluster is increased.
- After the loop, each cluster has total x, total y, and point count.

## 11. Recomputing centroid positions
- The next loop goes through each cluster.
- If a cluster has points, its new centroid is computed.
- `centroids[i].x = sumX[i] / count[i];` gives the average x.
- `centroids[i].y = sumY[i] / count[i];` gives the average y.
- This is the heart of K-Means: centroid = average of assigned points.

## 12. main function input
- `int n, k, iterations;` stores the user input values.
- The program asks for the number of data points.
- Then it asks for the number of clusters.
- Then it asks for the number of iterations.
- If the values are invalid, the program prints an error and exits.
- `k > n` is also invalid because you cannot have more clusters than points.

## 13. Creating the point list
- `vector<Point> original(n);` creates storage for the input points.
- This vector will hold the original dataset.

## 14. Choosing random or manual data
- `int choice;` stores whether the user wants random or manual data.
- If the user enters 1, random data is generated.
- If the user enters 0, the program asks for each point manually.
- This makes the program flexible for testing.

## 15. Random data generation
- `srand(time(0));` seeds the random number generator.
- The loop fills each point with random x and y values from 0 to 99.
- `original[i].cluster = 0;` initializes the cluster label.
- This gives a simple random 2D dataset.

## 16. Manual data input
- If random data is not chosen, the program asks for each point.
- The user enters x and y for every point.
- Each point is stored with cluster set to 0 first.
- This is useful when the user wants a custom dataset.

## 17. Printing points when the dataset is small
- If `n <= 20`, the program prints all points.
- This makes debugging easier.
- Large datasets are not printed because that would be too much output.

## 18. Thread count list
- `int thread_counts[] = {1, 2, 4};` tests the program with 1, 2, and 4 threads.
- This is used to compare performance across different thread counts.

## 19. Printing the header
- The program prints a title for the K-Means benchmark.
- It prints the dataset size, number of clusters, and iterations.
- This makes the output easy to understand.

## 20. Loop over thread counts
- `for (int t : thread_counts)` runs the algorithm several times.
- Each run uses a different OpenMP thread count.
- This helps show scaling behavior.

## 21. Copying the points
- `vector<Point> points = original;` makes a fresh copy for each run.
- This is important because K-Means changes the cluster values.
- Every thread-count test should start from the same data.

## 22. Initializing centroids
- `vector<Point> centroids(k);` stores the cluster centers.
- The code picks random points as starting centroids.
- Random starting points are common in K-Means.
- Different runs can start from different centroids.

## 23. Setting thread count
- `omp_set_num_threads(t);` tells OpenMP how many threads to use.
- This is the main control for the scaling test.

## 24. Printing initial centroids
- The program prints the starting centroids before K-Means runs.
- This helps the user see where clustering began.

## 25. Timing the algorithm
- `omp_get_wtime()` starts and stops the timer around `kmeans_parallel()`.
- `executionTime` stores the total runtime.
- `throughput` measures points processed per second.
- This is useful for performance comparison.

## 26. Printing final centroids
- After K-Means finishes, the program prints the final centroid locations.
- These values should be closer to the centers of the clusters in the data.
- That shows the algorithm has updated the means correctly.

## 27. Printing final cluster assignments
- If the dataset is small, the program prints each point's cluster label.
- This helps with debugging and learning.
- For large data, the labels are not printed because there are too many.

## 28. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This program teaches the K-Means clustering algorithm.
- OpenMP is used in the point-assignment step.
- The program also shows how cluster performance changes with thread count.