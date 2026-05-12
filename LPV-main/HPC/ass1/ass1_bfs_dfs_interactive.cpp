// Build: g++ ass1_bfs_dfs_interactive.cpp -o ass1_bfs_dfs_interactive -fopenmp
// Run: ./ass1_bfs_dfs_interactive

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

// Example Input:
// Enter number of vertices: 7
// Enter number of edges: 6
// Enter edge 1 (u v): 0 1
// Enter edge 2 (u v): 0 2
// Enter edge 3 (u v): 1 3
// Enter edge 4 (u v): 1 4
// Enter edge 5 (u v): 2 5
// Enter edge 6 (u v): 2 6
// Enter starting vertex: 0

// Expected Output:
// Sequential BFS: 0 1 2 3 4 5 6
// Parallel BFS: 0 1 2 3 4 5 6
// Sequential DFS: 0 1 3 4 2 5 6
// Parallel DFS: 0 1 3 4 2 5 6

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void printGraph() {
        cout << "\nGraph Representation\n";
        for (int i = 0; i < V; i++) {
            cout << "Vertex " << i << " -> ";
            for (int j : adj[i])
                cout << j << " ";
            cout << endl;
        }
    }

    void seqBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            cout << node << " ";

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            cout << node << " ";

#pragma omp parallel for num_threads(4)
            for (int i = 0; i < adj[node].size(); i++) {
                int neighbor = adj[node][i];

                if (!visited[neighbor]) {
#pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    void seqDFS(int start) {
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

            for (int i = adj[node].size() - 1; i >= 0; i--) {
                int neighbor = adj[node][i];
                if (!visited[neighbor])
                    st.push(neighbor);
            }
        }
    }

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

#pragma omp parallel for num_threads(4)
            for (int i = adj[node].size() - 1; i >= 0; i--) {
                int neighbor = adj[node][i];

                if (!visited[neighbor]) {
#pragma omp critical
                    {
                        if (!visited[neighbor])
                            st.push(neighbor);
                    }
                }
            }
        }
    }
};

int main() {
    int V, E;

    cout << "Enter number of vertices: ";
    cin >> V;

    cout << "Enter number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "\nEnter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "\nEnter starting vertex: ";
    cin >> start;

    if (start < 0 || start >= V) {
        cout << "Invalid starting vertex!\n";
        return 1;
    }

    omp_set_num_threads(4);

    cout << "\nUsing " << omp_get_max_threads() << " threads\n";

    g.printGraph();

    double start_time, end_time;

    cout << "\nSequential BFS: ";
    start_time = omp_get_wtime();
    g.seqBFS(start);
    end_time = omp_get_wtime();
    cout << "\nExecution Time: " << end_time - start_time << " seconds\n";

    cout << "\nParallel BFS: ";
    start_time = omp_get_wtime();
    g.parallelBFS(start);
    end_time = omp_get_wtime();
    cout << "\nExecution Time: " << end_time - start_time << " seconds\n";

    cout << "\nSequential DFS: ";
    start_time = omp_get_wtime();
    g.seqDFS(start);
    end_time = omp_get_wtime();
    cout << "\nExecution Time: " << end_time - start_time << " seconds\n";

    cout << "\nParallel DFS: ";
    start_time = omp_get_wtime();
    g.parallelDFS(start);
    end_time = omp_get_wtime();
    cout << "\nExecution Time: " << end_time - start_time << " seconds\n";

    return 0;
}
