// Build: g++ ass1_bfs_dfs.cpp -o ass1_bfs_dfs -fopenmp
// Run: ./ass1_bfs_dfs

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

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

    // ----------- SEQUENTIAL BFS -----------
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

    // ----------- PARALLEL BFS -----------
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

    // ----------- SEQUENTIAL DFS -----------
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

    // ----------- PARALLEL DFS -----------
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

    int V = 7;

    Graph g(V);

    // Programmatically added edges
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    int start = 0;

    omp_set_num_threads(4);

    cout << "Using " << omp_get_max_threads() << " threads\n";

    g.printGraph();

    double start_time, end_time;

    // ----------- SEQUENTIAL BFS -----------
    cout << "\nSequential BFS: ";

    start_time = omp_get_wtime();

    g.seqBFS(start);

    end_time = omp_get_wtime();

    cout << "\nExecution Time: "
         << end_time - start_time
         << " seconds\n";

    // ----------- PARALLEL BFS -----------
    cout << "\nParallel BFS: ";

    start_time = omp_get_wtime();

    g.parallelBFS(start);

    end_time = omp_get_wtime();

    cout << "\nExecution Time: "
         << end_time - start_time
         << " seconds\n";

    // ----------- SEQUENTIAL DFS -----------
    cout << "\nSequential DFS: ";

    start_time = omp_get_wtime();

    g.seqDFS(start);

    end_time = omp_get_wtime();

    cout << "\nExecution Time: "
         << end_time - start_time
         << " seconds\n";

    // ----------- PARALLEL DFS -----------
    cout << "\nParallel DFS: ";

    start_time = omp_get_wtime();

    g.parallelDFS(start);

    end_time = omp_get_wtime();

    cout << "\nExecution Time: "
         << end_time - start_time
         << " seconds\n";

    // ----------- PERFORMANCE COMPARISON -----------

    double seq_start, seq_end, par_start, par_end;

    seq_start = omp_get_wtime();
    g.seqBFS(start);
    seq_end = omp_get_wtime();

    par_start = omp_get_wtime();
    g.parallelBFS(start);
    par_end = omp_get_wtime();

    double seq_time = seq_end - seq_start;
    double par_time = par_end - par_start;

    double speedup = seq_time / par_time;
    double efficiency = speedup / 4;

    cout << "\n\n===== PERFORMANCE COMPARISON =====\n";

    cout << "Sequential BFS Time: "
         << seq_time << " seconds\n";

    cout << "Parallel BFS Time: "
         << par_time << " seconds\n";

    cout << "Speedup: "
         << speedup << endl;

    cout << "Efficiency: "
         << efficiency << endl;

    cout << "Throughput: "
         << 1 / par_time
         << " operations/sec\n";

    return 0;
}