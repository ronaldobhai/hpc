# Assignment 1: Parallel BFS and DFS - Code Explanation

## Overview
This program implements graph traversal algorithms (BFS and DFS) in both sequential and parallel versions using OpenMP.

---

## Header Files & Includes

```cpp
#include <iostream>       // For input/output operations
#include <vector>         // For dynamic arrays (adjacency list)
#include <queue>          // For FIFO queue structure (BFS)
#include <stack>          // For LIFO stack structure (DFS)
#include <omp.h>          // OpenMP for parallelization
```

---

## Graph Class

### Constructor
```cpp
Graph(int V) {
    this->V = V;              // Store number of vertices
    adj.resize(V);            // Resize adjacency list to V elements
}
```
- Initializes a graph with `V` vertices
- `adj` is a vector of vectors representing adjacency list (each vertex has a list of neighbors)

### addEdge Method
```cpp
void addEdge(int u, int v) {
    adj[u].push_back(v);      // Add v to u's adjacency list
    adj[v].push_back(u);      // Add u to v's adjacency list (undirected graph)
}
```
- Creates an undirected edge between vertices `u` and `v`
- Edge is added in both directions

### printGraph Method
```cpp
void printGraph() {
    cout << "\nGraph Representation\n";
    for (int i = 0; i < V; i++) {
        cout << "Vertex " << i << " -> ";
        for (int j : adj[i])
            cout << j << " ";
        cout << endl;
    }
}
```
- Prints the adjacency list representation of the graph
- Shows all neighbors for each vertex

---

## Sequential BFS (Breadth-First Search)

```cpp
void seqBFS(int start) {
    vector<bool> visited(V, false);    // Track visited vertices
    queue<int> q;                      // Create FIFO queue
    
    visited[start] = true;             // Mark start as visited
    q.push(start);                     // Add start to queue
    
    while (!q.empty()) {               // Process until queue is empty
        int node = q.front();          // Get first node in queue
        q.pop();                       // Remove it from queue
        
        cout << node << " ";           // Print visited node
        
        for (int neighbor : adj[node]) {    // Check all neighbors
            if (!visited[neighbor]) {       // If not visited
                visited[neighbor] = true;  // Mark as visited
                q.push(neighbor);          // Add to queue
            }
        }
    }
}
```
**Key Points:**
- Uses a queue for FIFO ordering
- Explores vertices level by level
- Time Complexity: O(V + E) where V is vertices, E is edges

---

## Parallel BFS

```cpp
void parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        cout << node << " ";
        
        #pragma omp parallel for num_threads(4)    // Parallelize neighbor exploration
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            
            if (!visited[neighbor]) {
                #pragma omp critical                // Critical section for safety
                {
                    if (!visited[neighbor]) {       // Double-check to avoid duplicates
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
}
```
**Parallelization Strategy:**
- `#pragma omp parallel for num_threads(4)`: Parallelize the loop over neighbors with 4 threads
- `#pragma omp critical`: Ensure only one thread at a time updates shared data
- Double-check pattern prevents race conditions

---

## Sequential DFS (Depth-First Search)

```cpp
void seqDFS(int start) {
    vector<bool> visited(V, false);    // Track visited vertices
    stack<int> st;                     // Create LIFO stack
    
    st.push(start);                    // Push start vertex
    
    while (!st.empty()) {              // Process until stack is empty
        int node = st.top();           // Get top of stack
        st.pop();                      // Remove from stack
        
        if (!visited[node]) {
            visited[node] = true;      // Mark as visited
            cout << node << " ";       // Print vertex
        }
        
        for (int i = adj[node].size() - 1; i >= 0; i--) {  // Iterate in reverse
            int neighbor = adj[node][i];
            if (!visited[neighbor])
                st.push(neighbor);     // Push unvisited neighbors
        }
    }
}
```
**Key Points:**
- Uses a stack for LIFO ordering
- Explores deeply before backtracking
- Iterates neighbors in reverse to maintain left-to-right order
- Time Complexity: O(V + E)

---

## Parallel DFS

```cpp
void parallelDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> st;
    
    st.push(start);
    
    while (!st.empty()) {
        int node = st.top();
        st.pop();
        
        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";
        }
        
        #pragma omp parallel for num_threads(4)    // Parallelize neighbor exploration
        for (int i = adj[node].size() - 1; i >= 0; i--) {
            int neighbor = adj[node][i];
            
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    if (!visited[neighbor])
                        st.push(neighbor);         // Push neighbors to stack
                }
            }
        }
    }
}
```
**Similar to Parallel BFS:**
- Parallelizes neighbor exploration with 4 threads
- Uses critical section for thread safety
- Double-check pattern prevents race conditions

---

## Main Function

```cpp
int main() {
    int V = 7;                         // Create graph with 7 vertices
    Graph g(V);
    
    // Add edges to create the graph
    g.addEdge(0, 1);  // 0-1
    g.addEdge(0, 2);  // 0-2
    g.addEdge(1, 3);  // 1-3
    g.addEdge(1, 4);  // 1-4
    g.addEdge(2, 5);  // 2-5
    g.addEdge(2, 6);  // 2-6
    
    int start = 0;    // Start from vertex 0
    
    omp_set_num_threads(4);            // Set number of threads
    cout << "Using " << omp_get_max_threads() << " threads\n";
    
    g.printGraph();                    // Display graph structure
    
    // ... Execution and timing of all four algorithms ...
    // ... Performance comparison metrics ...
}
```

---

## Performance Metrics Calculated

```cpp
double speedup = seq_time / par_time;           // How much faster parallel is
double efficiency = speedup / 4;                // Speedup per thread
double throughput = 1 / par_time;              // Operations per second
```

- **Speedup**: Ratio of sequential to parallel execution time
- **Efficiency**: How well threads are utilized (speedup / number of threads)
- **Throughput**: Number of operations completed per second

---

## Compilation & Execution

```bash
# Compile
g++ ass1_bfs_dfs.cpp -o ass1_bfs_dfs -fopenmp

# Run
./ass1_bfs_dfs
```

---

## Summary Table

| Aspect | Sequential BFS | Parallel BFS | Sequential DFS | Parallel DFS |
|--------|---|---|---|---|
| Data Structure | Queue (FIFO) | Queue (FIFO) | Stack (LIFO) | Stack (LIFO) |
| Order | Level-by-level | Level-by-level | Deep-first | Deep-first |
| Parallelization | None | 4 threads | None | 4 threads |
| Thread Safety | N/A | Critical section | N/A | Critical section |
| Time Complexity | O(V+E) | O(V+E) | O(V+E) | O(V+E) |
