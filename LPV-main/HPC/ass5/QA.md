# Assignment 5: K-Means Clustering - Questions & Answers

## Q1: What is K-Means clustering?

**Answer:**
K-Means is an iterative unsupervised learning algorithm that groups data into k clusters.

**Algorithm:**
1. **Initialize**: Choose k random points as initial centroids
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until convergence

**Goal**: Minimize within-cluster distance, maximize between-cluster distance

---

## Q2: What is a centroid?

**Answer:**
**Centroid**: The center (mean) of a cluster.

**Calculation:**
```
centroid_x = (sum of x-coordinates of all points) / (number of points)
centroid_y = (sum of y-coordinates of all points) / (number of points)
```

**Example:**
```
Points in cluster: (1,2), (3,4), (5,6)
Centroid = ((1+3+5)/3, (2+4+6)/3) = (3, 4)
```

**Represents**: Characteristic position of the cluster

---

## Q3: Why is Euclidean distance used?

**Answer:**
Euclidean distance measures straight-line distance in n-dimensional space.

**Formula:**
```
d = √[(x1-x2)² + (y1-y2)² + ... + (xn-xn)²]
```

**Advantages:**
- Intuitive (like distance on a map)
- Works in any dimension
- Minimized by mean (centroid definition)
- Differentiable (supports optimization)

**Alternatives:**
- Manhattan distance: |x1-x2| + |y1-y2|
- Cosine similarity: For sparse data
- Hamming distance: For categorical data

---

## Q4: How does the cluster assignment phase work?

**Answer:**
```cpp
#pragma omp parallel for
for(int i = 0; i < n; i++)
{
    float minDist = 1e9;           // Large initial value
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
    
    points[i].cluster = clusterIndex;  // Assign to nearest
}
```

**Process:**
1. For each point i:
   a. Initialize minimum distance to very large value
   b. For each centroid j:
      - Calculate distance from point to centroid
      - If closer than previous minimum, update minimum and cluster index
   c. Assign point to cluster with minimum distance

**Example:**
```
Point P(3, 4)
Centroids: C1(1, 1), C2(5, 5), C3(8, 8)

Distances:
  d(P, C1) = √[(3-1)² + (4-1)²] = √13 ≈ 3.6
  d(P, C2) = √[(3-5)² + (4-5)²] = √5 ≈ 2.2 ← Minimum!
  d(P, C3) = √[(3-8)² + (4-8)²] = √41 ≈ 6.4

Assign P to C2 (cluster 2)
```

**Why parallelize:**
- Each point assignment is independent
- No shared variable updates (only reads)
- Perfect for `#pragma omp parallel for`

---

## Q5: How does the centroid update phase work?

**Answer:**
```cpp
vector<float> sumX(k, 0), sumY(k, 0);  // Sums per cluster
vector<int> count(k, 0);               // Point counts per cluster

// Accumulate
for(int i = 0; i < n; i++) {
    int c = points[i].cluster;
    sumX[c] += points[i].x;
    sumY[c] += points[i].y;
    count[c]++;
}

// Recalculate
for(int i = 0; i < k; i++) {
    if(count[i] != 0) {
        centroids[i].x = sumX[i] / count[i];
        centroids[i].y = sumY[i] / count[i];
    }
}
```

**Process:**
1. For each point, add its x and y to its cluster's sum
2. For each cluster, divide sum by count to get mean
3. Move centroid to this new mean position

**Example:**
```
Cluster 0 points: (1,2), (2,3), (3,4)
Cluster 0 sums: sumX=6, sumY=9, count=3
New centroid: (6/3, 9/3) = (2, 3)
```

---

## Q6: What does `minDist = 1e9` do?

**Answer:**
`1e9` = 1,000,000,000 - a very large number used as initial value.

**Why?**
- Distance can't be negative
- Any calculated distance will be smaller than 1e9
- Guaranteed to find the actual minimum

**Example:**
```cpp
float minDist = 1e9;

if(dist < minDist)  // True because dist < 1,000,000,000
{
    minDist = dist;
    clusterIndex = j;
}
```

**Alternative approaches:**
```cpp
// Option 1: Initialize to first centroid distance
float minDist = distance(points[i], centroids[0]);
int clusterIndex = 0;

for(int j = 1; j < k; j++) {  // Start from 1
    ...
}

// Option 2: Use numeric limits
float minDist = numeric_limits<float>::max();
```

---

## Q7: What is the role of iterations?

**Answer:**
```cpp
for(int it = 0; it < iterations; it++)
{
    // Cluster assignment
    // Centroid update
}
```

**Purpose**: Run the algorithm for a fixed number of iterations.

**Why not run forever?**
- Computational cost: Each iteration takes O(n × k) time
- Convergence: Algorithm usually converges within 10-20 iterations
- Diminishing returns: Each iteration improves less and less

**Better approach (early stopping):**
```cpp
for(int it = 0; it < maxIterations; it++) {
    // Save old centroids
    vector<Point> oldCentroids = centroids;
    
    // Run iteration
    // ... assignment and update ...
    
    // Check convergence
    bool converged = true;
    for(int i = 0; i < k; i++) {
        if(distance(oldCentroids[i], centroids[i]) > 0.0001) {
            converged = false;
            break;
        }
    }
    
    if(converged) break;  // Stop early
}
```

---

## Q8: How does parallelization help K-Means?

**Answer:**
```cpp
#pragma omp parallel for
for(int i = 0; i < n; i++)  // Parallelize assignment
{
    // Find nearest centroid
    // ...
}
```

**Speedup sources:**
1. **Cluster assignment**: O(n) assignments parallelized
2. **Independent iterations**: Each point's assignment independent
3. **Thread scaling**: 4 threads → ~4x speedup (if n is large enough)

**Example with 100,000 points, 3 clusters:**
- Sequential: ~50 ms per assignment phase
- 4 threads: ~13 ms per assignment phase
- Speedup: 3.8x (good, but not perfect due to overhead)

**Where parallelization doesn't help:**
- Centroid update: Mostly sequential sums
- Small dataset: Overhead > benefit
- K very large: Inner loop already short

---

## Q9: What is throughput measurement?

**Answer:**
```cpp
double throughput = n / executionTime;  // Points per second
```

**Measurement:**
```
n = 100,000 points
executionTime = 0.5 seconds
throughput = 100,000 / 0.5 = 200,000 points/second
```

**Use cases:**
- Comparing algorithm performance
- Capacity planning (how many points/hour?)
- Scaling analysis (throughput vs threads)

**Typical values:**
```
1 thread:  100,000 points/sec
2 threads: 180,000 points/sec
4 threads: 320,000 points/sec
8 threads: 600,000 points/sec
```

---

## Q10: Why is the centroid update sequential in this implementation?

**Answer:**
```cpp
// Sequential accumulation
for(int i = 0; i < n; i++) {
    int c = points[i].cluster;
    sumX[c] += points[i].x;  // Multiple threads might update same cluster
    sumY[c] += points[i].y];
    count[c]++;
}
```

**Issue with parallelization:**
- Multiple threads may update sumX[c] simultaneously
- Race condition without protection
- Would need critical section or reduction

**Could be improved:**
```cpp
#pragma omp parallel for reduction(+:sumX[:k], sumY[:k], count[:k])
for(int i = 0; i < n; i++) {
    int c = points[i].cluster;
    sumX[c] += points[i].x;
    sumY[c] += points[i].y;
    count[c]++;
}
```

But current sequential version is simpler and centroid update is O(n), not O(n × k).

---

## Q11: What is convergence in K-Means?

**Answer:**
**Convergence**: Algorithm reaches a state where centroids stop moving significantly.

**Convergence indicators:**
1. Centroids don't change (or change < threshold)
2. Point assignments don't change
3. Objective function stabilizes

**Example:**
```
Iteration 1: Centroids change significantly
Iteration 5: Smaller changes
Iteration 10: Almost no change
Iteration 15: Converged (effectively no change)
```

**Mathematical definition:**
```
ε = max(distance(centroid[i]_old, centroid[i]_new) for all i)
If ε < threshold, converged!
```

---

## Q12: How does random centroid initialization affect results?

**Answer:**
Random initialization can cause different final clusters:

**Poor initialization:**
```
All 3 centroids start very close → Takes many iterations or wrong clusters
```

**Good initialization:**
```
Centroids spread across space → Faster convergence, better clusters
```

**K-means++ improvement:**
- First centroid: Pick any random point
- Next centroid: Pick point with probability ∝ distance² from nearest centroid
- Leads to faster convergence and better results

**Current implementation:**
```cpp
for(int i = 0; i < k; i++)
{
    centroids[i] = points[rand() % n];  // Random point
}
```

This is fine for simple demonstration but can be improved.

---

## Q13: What is the time complexity of K-Means?

**Answer:**
```
O(iterations × n × k)
```

Where:
- `iterations`: Number of iterations (typically 10-20)
- `n`: Number of points
- `k`: Number of clusters

**Example:**
```
iterations = 10
n = 100,000
k = 3

Total operations ≈ 10 × 100,000 × 3 = 3,000,000
```

**Complexity analysis per iteration:**
- Cluster assignment: O(n × k) - for each point, check k centroids
- Centroid update: O(n) - sum and divide
- Dominated by assignment: O(n × k)

---

## Q14: How does K-Means perform with different numbers of threads?

**Answer:**
Depends on problem size and data distribution:

**Ideal scaling (with 8 threads):**
```
1 thread:  1.0x baseline
2 threads: 1.95x (97.5% efficiency)
4 threads: 3.8x (95% efficiency)
8 threads: 7.5x (94% efficiency)
```

**Realistic scaling:**
```
1 thread:  1.0s
2 threads: 0.55s (1.8x speedup)
4 threads: 0.32s (3.1x speedup)
8 threads: 0.20s (5.0x speedup)
```

**Scaling limits:**
1. Assignment phase has O(n × k) work - parallelizes well
2. Update phase is O(n) - doesn't benefit as much
3. Overhead of thread management at some point exceeds benefit

**When to add more threads:**
- Large n (100k+)
- Large k (many clusters)
- Small iterations (overhead threshold high)

---

## Q15: What makes a good clustering result?

**Answer:**
**Quality metrics:**

1. **Silhouette Score**: Measures how similar points are to their cluster vs other clusters
   ```
   Range: [-1, 1]
   > 0.5: Good clustering
   ```

2. **Davies-Bouldin Index**: Average similarity between each cluster and its most similar cluster
   ```
   Lower is better
   < 1.0: Good clustering
   ```

3. **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
   ```
   Higher is better
   > 50: Generally good
   ```

**Visual inspection:**
- Points in same cluster close together
- Clusters well-separated
- No obvious outliers

---

## Q16: How would you handle the empty cluster problem?

**Answer:**
If a cluster has no points assigned:
```cpp
if(count[i] != 0) {
    centroids[i].x = sumX[i] / count[i];  // Skip if empty
    centroids[i].y = sumY[i] / count[i];
}
```

**Solutions:**
1. **Do nothing**: Leave centroid where it is (current approach)
2. **Reinitialize**: Pick a new random point
3. **Merge**: Combine empty cluster with another
4. **Better initialization**: K-means++ reduces empty clusters

---

## Q17: What is the difference between K-Means and K-Medoids?

**Answer:**

| Feature | K-Means | K-Medoids |
|---------|---------|-----------|
| **Centroid** | Mean of points | Actual data point |
| **Calculation** | Average | Median/mode |
| **Robustness** | Sensitive to outliers | Robust to outliers |
| **Computation** | Faster O(n×k) | Slower O(n²) |
| **Data type** | Numeric only | Any metric space |

**K-Means equation:**
```
New centroid = sum(points) / count
```

**K-Medoids:**
```
New medoid = point minimizing sum of distances to other points in cluster
```

---

## Q18: Can you parallelize the centroid update?

**Answer:**
Yes, with reduction:

```cpp
#pragma omp parallel for reduction(+:sumX[:k], sumY[:k], count[:k])
for(int i = 0; i < n; i++) {
    int c = points[i].cluster;
    sumX[c] += points[i].x;
    sumY[c] += points[i].y;
    count[c]++;
}
```

**Benefits:**
- Parallelizes reduction operations
- Better speedup if k is small

**Tradeoffs:**
- More complex code
- Memory overhead for private copies
- May not always be faster (depends on k)

---

## Q19: How would you visualize K-Means clusters?

**Answer:**
```cpp
// Write to file for plotting
ofstream outfile("clusters.txt");

for(int i = 0; i < n; i++) {
    outfile << points[i].x << " " 
            << points[i].y << " " 
            << points[i].cluster << "\n";
}

// In Python (matplotlib):
# import matplotlib.pyplot as plt
# import numpy as np
# 
# data = np.loadtxt("clusters.txt")
# x, y, cluster = data[:,0], data[:,1], data[:,2]
# 
# for c in range(3):
#     mask = cluster == c
#     plt.scatter(x[mask], y[mask], label=f"Cluster {c}")
# 
# plt.legend()
# plt.show()
```

---

## Q20: What are real-world applications of K-Means?

**Answer:**

1. **Customer Segmentation**: Group customers by purchase behavior
2. **Document Clustering**: Group similar documents/papers
3. **Image Compression**: Reduce colors by clustering similar colors (K=256 colors)
4. **Gene Sequencing**: Cluster DNA sequences
5. **Anomaly Detection**: Outliers are far from centroids
6. **Recommendation System**: Users with similar behavior in same cluster
7. **Network Intrusion Detection**: Cluster network traffic patterns
8. **Preprocessing**: Reduce data size by representing clusters with centroids

**Example - Image Compression:**
```
Original image: 1 million pixels with 256 colors each
K-Means with k=8: Find 8 representative colors
Replace each pixel with nearest cluster color
Result: 1 million pixels with 8 colors (3 bits per pixel instead of 8)
Compression ratio: 8x!
```

---

## Q21: Why is K-Means not always optimal?

**Answer:**
K-Means may converge to **local minimum**, not global optimum.

**Visual example:**
```
Initialization A:        Initialization B:
  X                           X
  X  XX                       X
X    XX                    XX  XX
      XX                      XX
     (Better clustering)   (Worse clustering)

Different initial centroids → Different final results!
```

**Solutions:**
1. **K-means++**: Smart initialization
2. **Multiple runs**: Try 10+ different random initializations
3. **Elbow method**: Choose k by looking at within-cluster variance
4. **Use domain knowledge**: Initialize centroids from known patterns

---

## Q22: How would you determine the optimal k?

**Answer:**
**Elbow Method:**
```
Within-cluster sum of squares (WCSS) vs k:

WCSS
  |     .
  |    ..
  |   ...
  |  .... ← Elbow (optimal k)
  | .....
  |......
  └────────── k
  1 2 3 4 5 6
```

**Code:**
```cpp
vector<double> wcss;

for(int k = 1; k <= 10; k++) {
    // Run K-means with this k
    // Calculate WCSS = sum of distances from points to centroids
    wcss.push_back(calculateWCSS(points, centroids));
}

// Find elbow (biggest drop in WCSS)
for(int k = 1; k < 9; k++) {
    double drop = wcss[k-1] - wcss[k];
    cout << "k=" << k << ": drop=" << drop << "\n";
}
```

**Interpretation:**
- Look for where decrease slows down (elbow)
- That k is usually optimal

---

## Q23: What is silhouette analysis?

**Answer:**
Measures how similar a point is to its own cluster vs other clusters.

**Silhouette coefficient for point i:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = average distance to points in same cluster
b(i) = average distance to points in nearest other cluster

Range: [-1, 1]
s(i) > 0: Point well-placed (closer to own cluster)
s(i) < 0: Point possibly misclassified
```

**Example:**
```
Point in well-defined cluster: s(i) ≈ 0.7
Point on boundary: s(i) ≈ 0.1
Point in wrong cluster: s(i) ≈ -0.3
```

---

## Q24: How to handle large-scale K-Means?

**Answer:**
For very large datasets:

1. **Mini-batch K-Means**: Use random subset each iteration
2. **Streaming K-Means**: Update centroids incrementally
3. **Distributed K-Means**: Spark, Hadoop implementation
4. **Approximate algorithms**: Skip some distance calculations

**Mini-batch example:**
```cpp
for(int it = 0; it < iterations; it++) {
    // Use only random 10% of points
    vector<int> indices;
    for(int i = 0; i < n/10; i++) {
        indices.push_back(rand() % n);
    }
    
    // Run assignment/update on subset
    // ... reduce update
}
```

**Speed vs accuracy trade**
