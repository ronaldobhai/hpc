# Assignment 1: BFS & DFS - Questions & Answers

## Q1: What is the difference between BFS and DFS?

**Answer:**
- **BFS (Breadth-First Search)**: Explores vertices level by level, uses a FIFO queue, explores all neighbors before moving deeper
- **DFS (Depth-First Search)**: Explores as far as possible along each branch, uses a LIFO stack, backtracks when no unvisited neighbors

**Analogy**: BFS is like expanding in circles from the start; DFS is like walking down a path until you hit a dead end, then backtracking.

---

## Q2: Why is the `visited` array necessary?

**Answer:**
The `visited` array prevents infinite loops and duplicate visits:
- Without it, the algorithm would revisit the same vertex multiple times
- In graphs with cycles, this would cause infinite recursion/loops
- It ensures each vertex is processed exactly once

---

## Q3: What does `#pragma omp critical` do?

**Answer:**
`#pragma omp critical` creates a critical section where only one thread can execute at a time:
- Ensures thread-safe access to shared variables (`visited`, queue/stack)
- Prevents race conditions when multiple threads try to update simultaneously
- Can cause performance overhead due to serialization

---

## Q4: Why use a double-check in the parallel version (`if (!visited[neighbor])` check twice)?

**Answer:**
The double-check pattern prevents a race condition:
1. Thread A checks: `if (!visited[neighbor])` → true, enters critical section
2. Thread B checks: `if (!visited[neighbor])` → true, waits for critical section
3. Thread A marks it as visited and adds to queue, exits critical section
4. Thread B enters critical section and checks again (finds it visited now)
5. Without the second check, Thread B would add a duplicate to the queue

---

## Q5: What is the time complexity of BFS and DFS?

**Answer:**
Both have **O(V + E)** time complexity:
- V = number of vertices
- E = number of edges
- Each vertex is visited once: O(V)
- Each edge is examined once: O(E)
- Total: O(V + E)

---

## Q6: Can parallel BFS be faster than sequential BFS?

**Answer:**
It depends on the graph structure:
- **For large graphs**: Yes, especially with many independent neighbors
- **For small graphs**: Often no, due to parallelization overhead
- **Critical section overhead**: The critical section serializes updates, limiting speedup
- **Thread overhead**: Creating/managing threads has overhead costs

---

## Q7: How many threads are used in this program?

**Answer:**
The program uses **4 threads**:
```cpp
#pragma omp parallel for num_threads(4)  // Explicitly set to 4
omp_set_num_threads(4);                  // Also set to 4
```
This is hardcoded. You could make it configurable by reading from command line.

---

## Q8: What would happen if we remove the critical section?

**Answer:**
- Race condition would occur
- Multiple threads could:
  - Mark the same vertex as visited simultaneously
  - Add duplicates to the queue/stack
  - Cause incorrect output order
  - Potentially crash the program

---

## Q9: Why iterate neighbors in reverse (backward loop) in DFS?

**Answer:**
In the DFS implementation:
```cpp
for (int i = adj[node].size() - 1; i >= 0; i--)  // Backward iteration
```
This maintains left-to-right order when using a stack (LIFO):
- Stack reverses order: last pushed = first popped
- By pushing in reverse, we pop in forward order
- Ensures consistent output between sequential and parallel versions

---

## Q10: What does `omp_get_wtime()` do?

**Answer:**
`omp_get_wtime()` returns the current wall-clock time in seconds:
```cpp
double start = omp_get_wtime();
// ... some code ...
double end = omp_get_wtime();
double time = end - start;  // Time elapsed
```
Used for performance measurements in the program.

---

## Q11: Can we use OpenMP for DFS on a recursive version?

**Answer:**
Yes, but it's more complex:
```cpp
void recursiveDFS(int node, vector<bool>& visited) {
    visited[node] = true;
    cout << node << " ";
    
    #pragma omp parallel for
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            recursiveDFS(neighbor, visited);  // Recursive call
        }
    }
}
```
However:
- Recursive parallelization is tricky with OpenMP
- Must use `#pragma omp task` for proper handling
- More overhead than iterative approach

---

## Q12: What are the limitations of this parallel implementation?

**Answer:**
1. **Critical section bottleneck**: All threads serialize at critical section
2. **Small graph overhead**: Parallelization cost exceeds benefit for small graphs
3. **Memory overhead**: 4 threads create context-switching overhead
4. **Load imbalance**: Some threads might finish before others if neighbors vary

---

## Q13: How would you measure speedup and efficiency?

**Answer:**
```cpp
double speedup = sequential_time / parallel_time;
double efficiency = speedup / number_of_threads;
double throughput = vertices_processed / parallel_time;
```
- **Speedup** should be close to 4 for ideal parallelization
- **Efficiency** should be close to 1 (or 100%)
- Values < 1 mean parallelization added more overhead than benefit

---

## Q14: What is the Graph structure used here?

**Answer:**
**Adjacency List representation**:
```
Vertex 0 -> [1, 2]
Vertex 1 -> [0, 3, 4]
Vertex 2 -> [0, 5, 6]
...
```
**Advantages**:
- Space efficient: O(V + E)
- Better for sparse graphs
- Easy neighbor iteration

**Alternative**: Adjacency Matrix (2D array, uses O(V²) space)

---

## Q15: When would you use BFS vs DFS?

**Answer:**

| Use Case | Algorithm | Reason |
|----------|-----------|--------|
| Shortest path (unweighted) | BFS | Finds shortest path naturally |
| Topological sort | DFS | Natural ordering |
| Connected components | Either | Both work equally well |
| Detecting cycles | Both | Can be used with different logic |
| Backtracking problems | DFS | Natural fit for exhaustive search |
| Collaborative filtering | BFS | Explores neighbors at each level |
| Maze solving | DFS | Often used with recursion |
| Network broadcasting | BFS | Explores level by level |

---

## Q16: How would you handle a directed graph instead of undirected?

**Answer:**
Change `addEdge` method:
```cpp
void addEdge(int u, int v) {
    adj[u].push_back(v);      // Only one direction
    // Remove: adj[v].push_back(u);
}
```
The traversal algorithms remain the same. The graph structure change handles directionality.

---

## Q17: What if we want to count the number of vertices visited?

**Answer:**
Add a counter:
```cpp
void seqBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    int count = 0;  // Counter
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        count++;  // Increment counter
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    
    cout << "Vertices visited: " << count << endl;
}
```

---

## Q18: How would parallelization work better for this algorithm?

**Answer:**
Better approaches than critical sections:
1. **Task parallelism**: Assign separate BFS fronts to different threads
2. **Level synchronization**: Process all vertices at one level in parallel, sync before next level
3. **Hybrid approach**: Use critical section only when necessary, avoid for read-only operations
4. **Work stealing**: Distribute work dynamically among threads

---

## Q19: What happens with multiple disconnected components?

**Answer:**
- BFS/DFS from one starting vertex will only visit connected vertices
- Other components remain unvisited
- To visit all vertices:
```cpp
for (int i = 0; i < V; i++) {
    if (!visited[i]) {
        seqBFS(i);  // Start new traversal from unvisited vertex
    }
}
```

---

## Q20: How would you modify this for weighted graphs?

**Answer:**
For weighted graphs, you'd typically use Dijkstra's algorithm (similar to BFS but with priority queue):
```cpp
priority_queue<pair<int, int>> pq;  // (distance, vertex)
vector<int> dist(V, INF);

dist[start] = 0;
pq.push({0, start});

while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    
    if (d > dist[u]) continue;
    
    for (auto [v, w] : adj[u]) {  // w is weight
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
    }
}
```

The graph representation would also need to store edge weights.
