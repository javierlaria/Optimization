import heapq
import networkx as nx
import matplotlib.pyplot as plt
import random

n = 8

matrix = [[0 if i == j or random.random() > 0.23 else random.randint(1, 10) for j in range(n)] for i in range(n)]
matrix[0][random.randint(1, n-1)] = random.randint(1, 10)
edges = [(i, j, matrix[i][j]) for i in range(n) for j in range(n) if matrix[i][j] != 0]

print("Adjacency Matrix:")
for row in matrix:
    print(row)

print("\nEdges with weights:")
print(edges)
distance_vector = [float("inf")] * len(matrix)
distance_vector[0] = 0 

def distance(matrix, i, j):
    return matrix[i][j] if matrix[i][j] != 0 else None

def neighbours(matrix, i):
    return [j for j in range(len(matrix)) if distance(matrix, i, j) is not None]

def dijkstra(matrix, distance_vector):

    visited = set()
    heap = [(0, 0)]  # (distance, vertex)
    previous = [None] * len(matrix)

    while heap:
        current_dist, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        print(f"\nVisiting vertex {u}, current distance: {current_dist}")
        
        for v in neighbours(matrix, u):
            if v in visited:
                continue
            alt = current_dist + distance(matrix, u, v)
            if alt < distance_vector[v]:
                distance_vector[v] = alt
                previous[v] = u
                heapq.heappush(heap, (alt, v))
                print(f" -> Updated distance to vertex {v}: {alt}")
    
    print("\nShortest paths from node 0:")
    for target in range(len(matrix)):
        if distance_vector[target] == float('inf'):
            print(f"Node {target} is unreachable from node 0")
            continue

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        print(f"To node {target}: path = {path}, distance = {distance_vector[target]}")
    return distance_vector

shortest_distances = dijkstra(matrix, distance_vector)
print("\nFinal shortest distances from source vertex 0:")
print(shortest_distances)

G = nx.MultiDiGraph()
G.add_weighted_edges_from(edges)
labels = nx.get_edge_attributes(G, "weight")
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.2', node_size = 1000, node_color = 'red')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, connectionstyle='arc3, rad = 0.2')
plt.show()

