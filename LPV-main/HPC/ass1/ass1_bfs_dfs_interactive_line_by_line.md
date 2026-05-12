# ass1_bfs_dfs_interactive.cpp - Line by Line Explanation

This file does the same BFS and DFS work as the original graph program, but now it asks the user for the graph input instead of using a fixed graph.

## 1. Build and run comments
- Line 1 shows the compile command with `g++` and `-fopenmp`.
- Line 2 shows how to run the program.

## 2. Library includes
- `#include <bits/stdc++.h>` includes most standard headers.
- `#include <iostream>` is for input and output.
- `#include <vector>` is for the adjacency list and visited arrays.
- `#include <queue>` is for BFS.
- `#include <stack>` is for DFS.
- `#include <omp.h>` is for OpenMP timing and parallel directives.

## 3. Example input comments
- The example input lines show the user what kind of numbers to enter.
- The expected output comments show the kind of traversal order the user should see.
- These comments are for learning and testing.

## 4. Namespace
- `using namespace std;` lets the code use standard names without the `std::` prefix.

## 5. Graph class members
- The `Graph` class keeps the graph data and traversal functions together.
- `int V;` stores the number of vertices.
- `vector<vector<int>> adj;` stores the neighbors of each vertex.

## 6. Graph constructor
- `Graph(int V)` creates a graph with `V` vertices.
- `this->V = V;` saves the number of vertices inside the object.
- `adj.resize(V);` makes an empty list for each vertex.

## 7. addEdge function
- `void addEdge(int u, int v)` adds an edge between two vertices.
- `adj[u].push_back(v);` stores `v` as a neighbor of `u`.
- `adj[v].push_back(u);` stores `u` as a neighbor of `v`.
- This makes the graph undirected.

## 8. printGraph function
- This function prints the full adjacency list.
- The outer loop visits each vertex from 0 to `V-1`.
- The inner loop prints all neighbors of that vertex.
- This helps the user confirm the graph was entered correctly.

## 9. Sequential BFS
- `void seqBFS(int start)` performs BFS in the usual single-thread way.
- `vector<bool> visited(V, false);` keeps track of which vertices were already seen.
- `queue<int> q;` holds the next nodes to visit.
- `visited[start] = true;` marks the start node as already discovered.
- `q.push(start);` puts the start node into the queue.
- The `while` loop keeps going until the queue becomes empty.
- `q.front()` reads the next node to visit.
- `q.pop()` removes it from the queue.
- `cout << node << " ";` prints the node.
- The neighbor loop checks all neighbors of the current node.
- A neighbor is added only if it has not been visited yet.
- This gives a level-by-level BFS order.

## 10. Parallel BFS
- `void parallelBFS(int start)` tries to speed up the neighbor checking step.
- A shared queue is still used, so the method is not a fully safe parallel BFS design.
- The start node is added to the queue first.
- The `while` loop removes one node at a time.
- `#pragma omp parallel for num_threads(4)` divides the neighbor loop across 4 threads.
- Each thread looks at one neighbor index.
- If the neighbor has not been visited, the code enters a `critical` section.
- The second visited check inside the critical block is needed because another thread may have already handled that neighbor.
- `visited[neighbor] = true;` marks the neighbor as discovered.
- `q.push(neighbor);` adds it to the queue for later processing.
- The output order can change because thread scheduling changes the push order.

## 11. Sequential DFS
- `void seqDFS(int start)` performs DFS with a stack.
- `stack<int> st;` stores nodes to visit.
- The start node is pushed first.
- The `while` loop continues until the stack is empty.
- `st.top()` gets the current node.
- `st.pop()` removes it.
- If the node has not been visited, it is marked and printed.
- The neighbor loop runs backward so the traversal order looks more like recursive DFS.
- Unvisited neighbors are pushed onto the stack.
- Since the stack is last-in first-out, the last pushed neighbor is visited first.

## 12. Parallel DFS
- `void parallelDFS(int start)` parallelizes the neighbor scan.
- It still uses one shared stack, so the order is not fixed and the logic is not a full lock-free parallel DFS.
- The current node is popped from the stack.
- If it was not visited, it is marked and printed.
- `#pragma omp parallel for num_threads(4)` splits the neighbor loop among threads.
- The critical section protects the shared stack when pushing new nodes.
- The extra visited check inside the critical section avoids duplicate pushes.
- Because the stack is shared, the final traversal order may be different from the sequential version.

## 13. main function input handling
- `int V, E;` stores the vertex count and edge count from the user.
- The program asks for the number of vertices first.
- Then it asks for the number of edges.
- `Graph g(V);` creates a graph with that many vertices.
- The program then asks for each edge one by one.
- `cin >> u >> v;` reads a pair of connected vertices.
- `g.addEdge(u, v);` stores that edge in the graph.
- Then it asks for the starting vertex.
- This makes the graph fully user controlled.

## 14. Input validation
- `if (start < 0 || start >= V)` checks that the start vertex is valid.
- If the start is outside the graph, the program prints an error message.
- `return 1;` ends the program early when the input is wrong.
- This prevents out-of-range array access.

## 15. OpenMP thread setup
- `omp_set_num_threads(4);` asks OpenMP to use 4 threads.
- `omp_get_max_threads()` prints the active maximum thread count.
- This helps the user know the parallel code is using OpenMP.

## 16. Printing the graph
- `g.printGraph();` displays the final graph structure.
- This confirms that the user input was read correctly.

## 17. Timing each traversal
- `double start_time, end_time;` are used to measure runtime.
- BFS and DFS are each timed separately.
- `omp_get_wtime()` gives a high precision wall clock time.
- The program prints the traversal order and then the execution time.
- This helps the user compare sequential and parallel versions.

## 18. Final return
- `return 0;` means the program finished without an error.

## Simple summary
- This version is the same graph traversal logic, but now the user gives the graph.
- It is better for testing because different graphs can be entered without changing the code.
- The parallel BFS and DFS sections are still approximate parallel versions because they share a queue or stack.