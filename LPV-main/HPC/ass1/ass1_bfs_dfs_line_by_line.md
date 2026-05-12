# ass1_bfs_dfs.cpp - Line by Line Explanation

This file shows how to do BFS and DFS on a small graph, and then compares sequential and parallel versions.

## 1. Build and run comments
- Line 1 tells you how to compile the file with `g++` and OpenMP support.
- Line 2 tells you how to run the built program.

## 2. Library includes
- `#include <bits/stdc++.h>` brings in almost all standard C++ headers at once.
- `#include <iostream>` is used for input and output.
- `#include <vector>` is used to store adjacency lists and visited flags.
- `#include <queue>` is used for BFS.
- `#include <stack>` is used for DFS.
- `#include <omp.h>` is used for OpenMP timing and parallel pragmas.

## 3. Namespace
- `using namespace std;` lets the code use `cout`, `vector`, `queue`, and other standard names without writing `std::` every time.

## 4. Graph class
- The `Graph` class stores the graph and all traversal functions.
- `int V;` stores the number of vertices.
- `vector<vector<int>> adj;` stores the adjacency list for each vertex.

## 5. Constructor
- `Graph(int V)` receives the number of vertices.
- `this->V = V;` saves the value in the class.
- `adj.resize(V);` creates one empty adjacency list for each vertex.

## 6. addEdge function
- `void addEdge(int u, int v)` adds an edge between two vertices.
- `adj[u].push_back(v);` adds `v` to `u`'s list.
- `adj[v].push_back(u);` adds `u` to `v`'s list because the graph is undirected.

## 7. printGraph function
- `void printGraph()` prints every vertex and its neighbors.
- The outer loop goes through each vertex.
- The inner loop prints every neighbor in that vertex's adjacency list.
- This helps you see the graph structure before traversal.

## 8. Sequential BFS
- `void seqBFS(int start)` performs normal BFS using one thread.
- `vector<bool> visited(V, false);` creates a visited array and marks all vertices as not visited.
- `queue<int> q;` creates the BFS queue.
- `visited[start] = true;` marks the start node as visited.
- `q.push(start);` puts the start node into the queue.
- The `while (!q.empty())` loop keeps running until there are no more nodes to visit.
- `int node = q.front();` takes the node at the front of the queue.
- `q.pop();` removes that node from the queue.
- `cout << node << " ";` prints the node.
- The `for` loop checks each neighbor of the current node.
- If a neighbor was not visited, it is marked visited and pushed into the queue.
- This is the standard BFS rule: visit by level.

## 9. Parallel BFS
- `void parallelBFS(int start)` tries to parallelize neighbor checking with OpenMP.
- It still uses one shared queue, so the algorithm is not a fully safe parallel BFS.
- `vector<bool> visited(V, false);` again tracks visited nodes.
- `queue<int> q;` stores the frontier.
- The start node is pushed first.
- The `while` loop processes nodes one by one.
- `#pragma omp parallel for num_threads(4)` splits the neighbor loop among 4 threads.
- Each thread looks at one neighbor at a time.
- The `if (!visited[neighbor])` check avoids pushing nodes that are already seen.
- `#pragma omp critical` protects the shared queue and visited update.
- The second `if (!visited[neighbor])` inside critical is important because another thread may have updated it first.
- `q.push(neighbor);` adds the new node to the queue safely.
- This gives some parallel work, but the queue order can change.

## 10. Sequential DFS
- `void seqDFS(int start)` does DFS using a stack.
- `vector<bool> visited(V, false);` marks all nodes unvisited first.
- `stack<int> st;` stores the DFS path.
- `st.push(start);` starts the traversal.
- The `while (!st.empty())` loop runs until the stack is empty.
- `int node = st.top();` reads the top node.
- `st.pop();` removes it.
- If the node was not visited, it is marked visited and printed.
- The neighbor loop goes from right to left using `adj[node].size() - 1`.
- Going backwards keeps the visit order closer to a typical recursive DFS.
- Unvisited neighbors are pushed onto the stack.
- Because it is a stack, the last neighbor pushed is visited first.

## 11. Parallel DFS
- `void parallelDFS(int start)` is the parallel version of DFS neighbor processing.
- It still uses one shared stack, so it is not a strict deterministic parallel DFS.
- The stack starts with the initial vertex.
- The `while` loop removes one current node at a time.
- If the node was not visited, it is printed and marked visited.
- `#pragma omp parallel for num_threads(4)` parallelizes the neighbor loop.
- Threads scan the neighbor list from back to front.
- The `critical` section protects the shared stack when pushing a neighbor.
- The second visited check inside the critical section prevents duplicates.
- Because threads race on the shared stack order, the output order may differ from sequential DFS.

## 12. main function setup
- `int V = 7;` hardcodes a graph with 7 vertices.
- `Graph g(V);` creates the graph object.
- The `addEdge` calls build the example graph.
- `int start = 0;` chooses vertex 0 as the start.
- `omp_set_num_threads(4);` asks OpenMP to use 4 threads.
- `cout << "Using " << omp_get_max_threads()` prints how many threads OpenMP will use.
- `g.printGraph();` shows the graph before traversal.

## 13. Timing the traversals
- `double start_time, end_time;` stores timing values.
- Each traversal is timed using `omp_get_wtime()`.
- The program prints sequential BFS, parallel BFS, sequential DFS, and parallel DFS.
- After each traversal, it prints the execution time in seconds.
- This helps compare the cost of the sequential and parallel versions.

## 14. Performance comparison section
- The final section runs BFS again to compare speeds.
- `seq_start` and `seq_end` measure sequential BFS time.
- `par_start` and `par_end` measure parallel BFS time.
- `double speedup = seq_time / par_time;` computes how much faster the parallel version is.
- `double efficiency = speedup / 4;` shows how well 4 threads were used.
- `double throughput = V / par_time;` measures how many vertices per second were processed.
- This section is mostly for analysis and may show small or even worse speedup for tiny graphs because thread overhead is high.

## 15. Final return
- `return 0;` ends the program successfully.

## Simple summary
- This file teaches graph traversal.
- BFS uses a queue.
- DFS uses a stack.
- OpenMP is used to parallelize neighbor processing, but the traversal order is not fully deterministic because the data structures are shared.