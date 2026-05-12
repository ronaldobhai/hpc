# ass5_kmeans_interactive.cpp - Line by Line Explanation

This is the user-input version of the K-Means program. It lets the user choose the dataset size, number of clusters, number of iterations, and whether the data is random or manual.

## 1. Build and run comments
- Line 1 shows the compile command with OpenMP and the math library.
- Line 2 shows how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard C++ headers.
- `#include <iostream>` is used for input and output.
- `#include <vector>` stores points and centroids.
- `#include <cmath>` provides `sqrt()`.
- `#include <cstdlib>` provides `rand()` and `srand()`.
- `#include <ctime>` provides `time(0)`.
- `#include <omp.h>` is used for parallelism and timing.

## 3. Example comments
- The example input shows one simple test case.
- The expected output comment tells the user what kind of output to expect.
- These comments make the file easier to test.

## 4. Namespace
- `using namespace std;` lets the program use standard names directly.

## 5. Point structure
- `struct Point` stores the x coordinate, y coordinate, and cluster number.
- This keeps the code simple and readable.

## 6. distance function
- `float distance(Point a, Point b)` computes the Euclidean distance between two points.
- It uses the standard distance formula.
- Smaller distance means the point is closer to the centroid.

## 7. kmeans_parallel function header
- This function does the main K-Means work.
- It receives the points, centroids, cluster count, and iteration count.
- `int n = points.size();` stores the number of points.

## 8. Main iteration loop
- The outer loop repeats the K-Means steps many times.
- Each iteration first assigns points to clusters and then updates centroids.
- Repeating this helps the centroids move toward the real cluster centers.

## 9. Point assignment with OpenMP
- `#pragma omp parallel for` splits the point loop across threads.
- Each thread handles some points independently.
- For each point, the code compares it to every centroid.
- The nearest centroid becomes the point's cluster label.
- `points[i].cluster = clusterIndex;` stores that label.
- This is a good parallel step because each point can be assigned separately.

## 10. Centroid update helpers
- `sumX`, `sumY`, and `count` store cluster totals.
- `sumX[c]` is the total x value for cluster `c`.
- `sumY[c]` is the total y value for cluster `c`.
- `count[c]` stores how many points belong to cluster `c`.
- These arrays are used to compute the new centroids.

## 11. Updating centroids
- The program scans every point after assignment.
- It adds each point's coordinates to the correct cluster totals.
- It also increases the point count for that cluster.
- Then it loops over all clusters.
- For each cluster with points, it computes the average x and average y.
- That average becomes the new centroid position.

## 12. Reading user input
- `int n, k, iterations;` stores the user choices.
- The program asks for the number of data points.
- Then it asks for the number of clusters.
- Then it asks for the number of iterations.
- If the values are invalid, the program stops with an error.
- `k > n` is rejected because more clusters than points does not make sense.

## 13. Creating the point array
- `vector<Point> original(n);` makes storage for the input points.
- This vector keeps the original data so it can be copied for each thread test.

## 14. Choosing random or manual data
- `int choice;` stores whether random data will be generated.
- If the user enters 1, the code creates random points.
- If the user enters 0, the user types each point manually.
- This makes the program flexible for testing.

## 15. Random data generation
- `srand(time(0));` seeds the random generator.
- The loop assigns random x and y values from 0 to 99.
- Each point's cluster is set to 0 first.
- The program prints that it is generating random data.

## 16. Manual data entry
- If random data is not chosen, the program asks for each point.
- The user enters x and y for every point.
- The cluster is set to 0 for each point at first.
- This is useful if the user wants a specific test set.

## 17. Printing small datasets
- If the dataset has 20 points or fewer, all points are printed.
- This is helpful for debugging and learning.
- Larger datasets are not printed because that would be too long.

## 18. Thread count list
- `int thread_counts[] = {1, 2, 4};` tells the program which thread counts to test.
- This is used to compare performance with different parallel settings.

## 19. Printing the benchmark header
- The program prints the title and basic test information.
- It shows the dataset size, cluster count, and iteration count.
- This makes the output easier to read.

## 20. Running the scaling test
- The `for (int t : thread_counts)` loop repeats the whole algorithm for each thread count.
- For each run, the original data is copied again.
- That makes each test fair.

## 21. Initial centroid selection
- `vector<Point> centroids(k);` stores the centroids.
- The code picks random points as the initial centroids.
- K-Means usually starts with random initial centroids.
- Different thread-count tests can therefore start from different initial guesses.

## 22. Setting OpenMP threads
- `omp_set_num_threads(t);` tells OpenMP how many threads to use for this run.
- This is the main setting that changes in the scaling test.

## 23. Printing initial centroids
- The code prints each starting centroid.
- This helps the user see where the algorithm begins.

## 24. Timing K-Means
- The timer starts before `kmeans_parallel()` and ends after it.
- `executionTime` stores the elapsed time.
- `throughput` is calculated as points per second.
- This shows how fast the clustering ran.

## 25. Printing final centroids
- After the algorithm finishes, the final centroid coordinates are printed.
- These are the learned cluster centers.
- If K-Means worked, the centroids should settle near dense groups of points.

## 26. Printing cluster assignments for small data
- If the dataset is small, the program prints the cluster label for each point.
- This lets the user check that the grouping makes sense.
- For larger datasets, the labels are skipped to avoid too much output.

## 27. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This file is the interactive K-Means demo.
- The user controls the input data and clustering settings.
- The program shows how K-Means behaves with different OpenMP thread counts.