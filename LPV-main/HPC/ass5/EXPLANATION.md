# Assignment 5: Parallel K-Means Clustering - Code Explanation

## Overview
This program implements the K-Means clustering algorithm using OpenMP parallelization, demonstrating thread scaling and performance metrics for machine learning workloads on HPC systems.

---

## Point Structure

```cpp
struct Point
{
    float x, y;        // 2D coordinates
    int cluster;       // Assigned cluster ID
};
```

**Purpose**: Represents a data point in 2D space with cluster assignment.

---

## Distance Function

```cpp
float distance(Point a, Point b)
{
    return sqrt(
        (a.x - b.x) * (a.x - b.x) +
        (a.y - b.y) * (a.y - b.y)
    );
}
```

**Calculates Euclidean distance:**
```
d = √[(x1-x2)² + (y1-y2)²]
```

**How it works:**
1. Calculate difference in x: (a.x - b.x)
2. Calculate difference in y: (a.y - b.y)
3. Square both differences
4. Sum the squares
5. Take square root

**Visualization:**
```
Point A (2, 3)
    │\
    │ \ distance = √[(4-2)² + (1-3)²] = √[4+4] = √8 ≈ 2.83
    │  \
Point B (4, 1)
```

---

## K-Means Main Algorithm

```cpp
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
            float minDist = 1e9;           // Large initial value
            int clusterIndex = 0;
            
            // Find nearest centroid
            for(int j = 0; j < k; j++)
            {
                float dist = distance(points[i], centroids[j]);
                
                if(dist < minDist)
                {
                    minDist = dist;
                    clusterIndex = j;
                }
            }
            
            points[i].cluster = clusterIndex;  // Assign to nearest cluster
        }
        
        /* ================= UPDATE CENTROIDS ================= */
        
        vector<float> sumX(k, 0);         // Sum of x-coordinates per cluster
        vector<float> sumY(k, 0);         // Sum of y-coordinates per cluster
        vector<int> count(k, 0);          // Point count per cluster
        
        // Accumulate points for each cluster
        for(int i = 0; i < n; i++)
        {
            int c = points[i].cluster;
            
            sumX[c] += points[i].x;        // Add x to cluster sum
            sumY[c] += points[i].y;        // Add y to cluster sum
            count[c]++;                    // Increment cluster count
        }
        
        // Recalculate centroids as mean of points in cluster
        for(int i = 0; i < k; i++)
        {
            if(count[i] != 0)              // Avoid division by zero
            {
                centroids[i].x = sumX[i] / count[i];   // Mean x
                centroids[i].y = sumY[i] / count[i];   // Mean y
            }
        }
    }
}
```

---

## K-Means Algorithm Explanation

**K-Means is an iterative clustering algorithm:**

### Iteration Process:

1. **Cluster Assignment Phase:**
   - For each point, find the nearest centroid
   - Assign point to that cluster

2. **Centroid Update Phase:**
   - For each cluster, calculate the mean of all assigned points
   - Move centroid to this new mean position

3. **Repeat:** Steps 1-2 until convergence (fixed number of iterations)

**Visual Example:**
```
Iteration 0:
  Initial random centroids: C1(25,75), C2(50,50), C3(75,25)
  
Iteration 1:
  - Assign each point to nearest centroid
  - Recalculate centroids based on assigned points
  - Centroids may move: C1(28,72), C2(52,48), C3(73,27)
  
Iteration 2-10:
  - Centroids converge to stable positions
  - Point assignments may stabilize
```

---

## Cluster Assignment Phase

```cpp
#pragma omp parallel for
for(int i = 0; i < n; i++)
{
    float minDist = 1e9;       // Initialize to large value
    int clusterIndex = 0;
    
    // Check distance to all centroids
    for(int j = 0; j < k; j++)
    {
        float dist = distance(points[i], centroids[j]);
        
        if(dist < minDist)
        {
            minDist = dist;
            clusterIndex = j;
        }
    }
    
    points[i].cluster = clusterIndex;  // Assign to closest cluster
}
```

**Parallelization:**
- `#pragma omp parallel for`: Each thread processes multiple points
- Independent iterations: No race condition (each point assigned individually)
- No critical sections needed

**Time Complexity**: O(n × k) where n = points, k = clusters

---

## Centroid Update Phase

```cpp
vector<float> sumX(k, 0);    // Accumulator for x-coordinates
vector<float> sumY(k, 0);    // Accumulator for y-coordinates
vector<int> count(k, 0);     // Count points per cluster

// Sequential accumulation
for(int i = 0; i < n; i++)
{
    int c = points[i].cluster;
    
    sumX[c] += points[i].x;   // Add to cluster's x sum
    sumY[c] += points[i].y;   // Add to cluster's y sum
    count[c]++;               // Increment cluster's point count
}

// Recalculate centroids
for(int i = 0; i < k; i++)
{
    if(count[i] != 0)
    {
        centroids[i].x = sumX[i] / count[i];  // New x = sum / count
        centroids[i].y = sumY[i] / count[i];  // New y = sum / count
    }
}
```

**Why sequential?**
- Accumulation requires reduction (could be parallel)
- Current implementation is simple and effective
- Could be optimized with `#pragma omp parallel for reduction`

---

## Main Function

### Step 1: Configuration
```cpp
int n = 100000;          // 100,000 data points
int k = 3;               // 3 clusters
int iterations = 10;     // 10 iterations
```

**Why these values?**
- **n=100,000**: Realistic dataset size for benchmarking
- **k=3**: Simple visualization and understanding
- **iterations=10**: Usually converges by then

---

### Step 2: Generate Random Data
```cpp
vector<Point> original(n);

srand(time(0));          // Seed for reproducibility

for(int i = 0; i < n; i++)
{
    original[i].x = rand() % 100;   // x in [0, 100)
    original[i].y = rand() % 100;   // y in [0, 100)
    original[i].cluster = 0;        // Initial cluster
}
```

**Data characteristics:**
- Uniform random distribution in 100×100 space
- All points initially unassigned (cluster 0)

---

### Step 3: Thread Scaling Experiment
```cpp
int thread_counts[] = {1, 2, 4, 8};  // Test with different thread counts

for(int t : thread_counts)
{
    vector<Point> points = original;  // Copy original data
    
    vector<Point> centroids(k);
    
    // Initialize centroids randomly from data
    for(int i = 0; i < k; i++)
    {
        centroids[i] = points[rand() % n];  // Pick random point
    }
    
    omp_set_num_threads(t);           // Set number of threads
    
    double start = omp_get_wtime();   // Start timer
    
    kmeans_parallel(points, centroids, k, iterations);  // Run algorithm
    
    double end = omp_get_wtime();     // End timer
    
    double executionTime = end - start;
    double throughput = n / executionTime;  // Points processed per second
    
    // Print results for this thread count
    cout << "\nThreads Used : " << t;
    cout << "\nExecution Time : " << executionTime << " seconds";
    cout << "\nThroughput : " << throughput << " points/sec\n";
}
```

**Measurements:**
- **Execution Time**: How long the algorithm takes
- **Throughput**: Number of point-cluster assignments per second
- **Speedup**: Can be calculated from these times

---

## Performance Metrics

### Throughput Calculation
```cpp
double throughput = n / executionTime;  // Points per second
```

**Example:**
- n = 100,000 points
- executionTime = 0.5 seconds
- throughput = 100,000 / 0.5 = 200,000 points/sec

### Speedup Calculation
```cpp
double speedup_1thread = time_1 / time_8;
```

**Example:**
- 1 thread: 1.0 second
- 8 threads: 0.15 seconds
- Speedup: 1.0 / 0.15 = 6.67x

### Efficiency
```cpp
double efficiency = speedup / num_threads;
```

**Example with 8 threads:**
- Speedup = 6.67x
- Efficiency = 6.67 / 8 = 83.4% (very good!)

---

## Why K-Means is Important for ML/HPC

1. **Common algorithm**: Used in data science, clustering, segmentation
2. **Parallelizable**: Main loop is embarrassingly parallel
3. **Scalable**: Works on large datasets (millions of points)
4. **Real-world use**: Image compression, customer segmentation, etc.
5. **Performance sensitive**: Speed critical for large data

---

## K-Means Complexity Analysis

| Operation | Time | Notes |
|-----------|------|-------|
| **Cluster assignment** | O(n × k) | n points × k distance calculations |
| **Centroid update** | O(n) | Sum all points per cluster |
| **Per iteration** | O(n × k) | Assignment dominates |
| **Total** | O(iterations × n × k) | Usually iterations < 20 |

**Example:**
- n = 100,000 points
- k = 3 clusters
- iterations = 10
- Total ops: 10 × 100,000 × 3 = 3,000,000 operations

---

## Initialization Strategy

```cpp
for(int i = 0; i < k; i++)
{
    centroids[i] = points[rand() % n];  // Pick random point
}
```

**Alternatives:**
1. **K-means++**: Smart selection reduces iterations needed
2. **Random centroids**: Simple but may need more iterations
3. **Uniform grid**: Divide space into k regions
4. **Preprocessed**: Use domain knowledge

---

## Convergence

The algorithm stops after fixed iterations. Better stopping criteria:

```cpp
// Check if centroids changed significantly
bool converged = true;
for(int i = 0; i < k; i++) {
    if(fabs(centroids_new[i].x - centroids_old[i].x) > 0.0001)
        converged = false;
}

if(converged) break;  // Stop early if converged
```

---

## Compilation & Execution

```bash
# Compile
g++ ass5.cpp -o ass5 -fopenmp -lm

# Run
./ass5
```

**Note**: `-lm` flag links math library for `sqrt()`

---

## Summary Table

| Aspect | Value | Notes |
|--------|-------|-------|
| **Data points** | 100,000 | Realistic size |
| **Clusters** | 3 | Small for visualization |
| **Dimensions** | 2 | 2D space |
| **Iterations** | 10 | Fixed number |
| **Thread counts** | 1, 2, 4, 8 | Scaling test |
| **Algorithm** | K-Means | Classic clustering |
| **Parallelization** | OpenMP | Shared memory |

---

## Real-World Applications

1. **Image Compression**: Reduce colors by clustering similar colors
2. **Customer Segmentation**: Group customers by behavior
3. **Document Clustering**: Group similar documents
4. **Gene Sequencing**: Cluster DNA sequences
5. **Recommendation Systems**: Cluster users with similar preferences
6. **Anomaly Detection**: Points far from centroids are outliers

---

## Advanced Topics

### Vector Quantization
K-means finds k representative vectors (centroids) that best represent the data.

### Lloyd's Algorithm
This is the classic K-means, also called Lloyd's algorithm.

### K-means++ Initialization
Improved initialization that often reduces iterations needed:
1. Choose first centroid randomly
2. For each remaining centroid, choose point with probability proportional to its distance² from nearest centroid
3. Run standard K-means

This often leads to better final clustering and faster convergence.
