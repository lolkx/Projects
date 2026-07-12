from vrp_objects import Node, Edge, Route, Solution
import numpy as np
import math
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import networkx as nx
import matplotlib.colors as mcolors


start_time = time.time()

def read_vrp_instance(filename):
    Node_list = []
    with open(filename, "r") as instance:
        for index, line in enumerate(instance):
            line = line.strip()
            if not line: continue
            try:
                x, y, demand = (float(value) for value in line.split())
            except ValueError:
                raise ValueError(f"Error en la línea {index+1}: '{line}'")
            node = Node(index, x, y, demand)
            Node_list.append(node)
    return Node_list

def compute_cost(inode, jnode):
    return math.sqrt((inode.x-jnode.x)**2 + (inode.y-jnode.y)**2)

def get_depot_edge(route, node, depot):
    origin = route.edges[0].origin
    end = route.edges[0].end
    if ((origin == node and end == depot) or (origin == depot and end == node)):
        return route.edges[0]
    else:
        return route.edges[-1]

def run_savings_on_cluster(cluster_nodes, depot, vehcap):
    """Applies Clarke-Wright Savings to a specific subset of nodes"""
    if not cluster_nodes:
        return []

    nodes_in_heuristic = [depot] + cluster_nodes
    n_cluster = len(nodes_in_heuristic)

    # Initialize basic info for nodes relative to depot
    for node in cluster_nodes:
        cost = compute_cost(depot, node)
        node.dnEdge = Edge(depot, node, cost)
        node.ndEdge = Edge(node, depot, cost)
        node.dnEdge.invEdge = node.ndEdge
        node.ndEdge.invEdge = node.dnEdge

    # Build Savings List
    cluster_savings = []
    for i in range(1, n_cluster):
        inode = nodes_in_heuristic[i]
        for j in range(i+1, n_cluster):
            jnode = nodes_in_heuristic[j]
            cost = compute_cost(inode, jnode)
            ijEdge = Edge(inode, jnode, cost)
            jiEdge = Edge(jnode, inode, cost)
            ijEdge.invEdge = jiEdge
            jiEdge.invEdge = ijEdge
            ijEdge.savings = inode.dnEdge.cost + jnode.dnEdge.cost - cost
            cluster_savings.append(ijEdge)

    cluster_savings.sort(key=lambda edge: edge.savings, reverse=True)

    # Initial Dummy Routes
    routes = []
    for node in cluster_nodes:
        dndRoute = Route()
        dndRoute.edges = [node.dnEdge, node.ndEdge]
        dndRoute.cost = node.dnEdge.cost + node.ndEdge.cost
        dndRoute.demand = node.demand
        node.inroute = dndRoute
        node.isInterior = False
        routes.append(dndRoute)

    # Merge Process
    for edge in cluster_savings:
        inode, jnode = edge.origin, edge.end
        iroute, jroute = inode.inroute, jnode.inroute

        # Check constraints
        if iroute == jroute or inode.isInterior or jnode.isInterior:
            continue
        if iroute.demand + jroute.demand > vehcap:
            continue

        # Merge logic
        iEdge = get_depot_edge(iroute, inode, depot)
        iroute.edges.remove(iEdge)
        iroute.cost -= iEdge.cost
        if len(iroute.edges) > 1: inode.isInterior = True
        if iroute.edges[0].origin != depot: iroute.reverse()

        jEdge = get_depot_edge(jroute, jnode, depot)
        jroute.edges.remove(jEdge)
        jroute.cost -= jEdge.cost
        if len(jroute.edges) > 1: jnode.isInterior = True
        if jroute.edges[0].origin == depot: jroute.reverse()

        iroute.edges.append(edge)
        iroute.cost += edge.cost
        iroute.demand += jnode.demand

        for e in jroute.edges:
            iroute.edges.append(e)
            iroute.cost += e.cost
            iroute.demand += e.end.demand if e.end != depot else 0
            if e.end != depot: e.end.inroute = iroute
            if e.origin != depot: e.origin.inroute = iroute

        routes.remove(jroute)

    return routes

# --- Main Execution ---
""" DICTIONARY OF INSTANCE NAMES AND ASSOCIATED VEHICLE CAPACITIES """
instances = {'A-n32-k5': 100, 'A-n38-k5': 100, 'A-n45-k7': 100, 'A-n55-k9': 100,
             'A-n60-k9': 100, 'A-n61-k9': 100, 'A-n65-k9': 100, 'A-n80-k10': 100, 'B-n50-k7': 100,
             'B-n52-k7': 100, 'B-n57-k9': 100, 'B-n78-k10': 100, 'E-n22-k4': 6000, 'E-n30-k3': 4500,
             'E-n33-k4': 8000, 'E-n51-k5': 160, 'E-n76-k7': 220, 'E-n76-k10': 140, 'E-n76-k14': 200,
             'F-n45-k4': 2010, 'F-n72-k4': 30000, 'F-n135-k7': 2210, 'M-n101-k10': 200, 'M-n121-k7': 200,
             'P-n22-k8': 3000, 'P-n40-k5': 140, 'P-n50-k10': 100, 'P-n55-k15': 70, 'P-n65-k10': 130,
             'P-n70-k10': 135, 'P-n76-k4': 350, 'P-n76-k5': 280, 'P-n101-k4': 400}


filename = 'A-n80-k10_input_nodes.txt'
instance = 'A-n80-k10'

Node_list = read_vrp_instance(filename)
depot = Node_list[0]
vehcap = instances[instance]

# 1. Prepare data for DBSCAN (Exclude depot)
coords = np.array([[n.x, n.y] for n in Node_list[1:]])

# 2. Apply DBSCAN
# eps: distance to neighbors, min_samples: min points to form a cluster
# These parameters may need tuning based on your coordinate scale
db = DBSCAN(eps=15, min_samples=2).fit(coords)
labels = db.labels_

# 3. Organize nodes into clusters
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters: clusters[label] = []
    clusters[label].append(Node_list[i+1])

# 4. Solve Savings for each cluster
sol = Solution()
for label, cluster_nodes in clusters.items():
    cluster_routes = run_savings_on_cluster(cluster_nodes, depot, vehcap)
    for r in cluster_routes:
        sol.routes.append(r)
        sol.cost += r.cost
        sol.demand += r.demand

# --- Output ---
for route in sol.routes:
    print(route)

print(f"Total Cost: {sol.cost:.2f}")
print(f"Execution Time: {time.time() - start_time:.4f}s")



def plot_clusters_and_solution(nodes, clusters, sol, instanceName):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Color palette for clusters
    palette = list(mcolors.TABLEAU_COLORS.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    depot = nodes[0]

    # --- LEFT: DBSCAN Clusters ---
    ax1.scatter(depot.x, depot.y, c='red', marker='s', s=120, label='Depot')

    cluster_color_map = {}
    for i, (label, cluster_nodes) in enumerate(clusters.items()):
        color = 'black' if label == -1 else palette[i % len(palette)]
        cluster_color_map[label] = color

        xs = [n.x for n in cluster_nodes]
        ys = [n.y for n in cluster_nodes]
        ax1.scatter(xs, ys, color=color, label=f"Cluster {label}" if label != -1 else "Noise",
                    edgecolors='k', s=60)
        for n in cluster_nodes:
            ax1.text(n.x + 0.5, n.y + 0.5, str(int(n.ID)), fontsize=8)

    ax1.set_title(f"DBSCAN Clusters: {instanceName}", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best')

    # --- RIGHT: VRP Routes ---
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.ID, coord=(node.x, node.y))

    # Draw edges for each route with cluster color
    for label, cluster_nodes in clusters.items():
        route_color = cluster_color_map[label]
        for route in sol.routes:
            # Determine if this route belongs to this cluster
            route_nodes = [e.origin for e in route.edges if e.origin != depot] + \
                          [e.end for e in route.edges if e.end != depot]
            if any(n in cluster_nodes for n in route_nodes):
                for edge in route.edges:
                    G.add_edge(edge.origin.ID, edge.end.ID, color=route_color)

    coord = nx.get_node_attributes(G, 'coord')
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, coord, ax=ax2, node_size=60, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, coord, ax=ax2, font_size=8)
    nx.draw_networkx_edges(G, coord, ax=ax2, edge_color=edge_colors, width=1.8)

    ax2.set_title(f"VRP Solution (Clustered): {instanceName}", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_clusters_and_solution(Node_list, clusters, sol, instance)

score = silhouette_score(coords, labels)

print(score)


def plot_instance_row(nodes, clusters, sol, instanceName, axes_row):
    """
    Plots one instance into a specific row of the 3x2 grid.
    axes_row: a list of two axes [ax_cluster, ax_solution]
    """
    palette = list(mcolors.TABLEAU_COLORS.values())
    depot = nodes[0]
    ax1, ax2 = axes_row  # Unpack the two axes for this row

    # --- LEFT: DBSCAN Clusters ---
    ax1.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')

    cluster_color_map = {}
    for i, (label, cluster_nodes) in enumerate(clusters.items()):
        color = 'black' if label == -1 else palette[i % len(palette)]
        cluster_color_map[label] = color

        xs = [n.x for n in cluster_nodes]
        ys = [n.y for n in cluster_nodes]
        ax1.scatter(xs, ys, color=color, edgecolors='k', s=50)
        for n in cluster_nodes:
            ax1.text(n.x + 0.4, n.y + 0.4, str(int(n.ID)), fontsize=7)

    ax1.set_ylabel(instanceName, fontsize=12, fontweight='bold') # Label the row
    ax1.set_title(f"DBSCAN Clusters", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- RIGHT: VRP Routes ---
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.ID, coord=(node.x, node.y))

    for label, cluster_nodes in clusters.items():
        route_color = cluster_color_map[label]
        for route in sol.routes:
            route_nodes = [e.origin for e in route.edges if e.origin != depot] + \
                          [e.end for e in route.edges if e.end != depot]
            if any(n in cluster_nodes for n in route_nodes):
                for edge in route.edges:
                    G.add_edge(edge.origin.ID, edge.end.ID, color=route_color)

    coord = nx.get_node_attributes(G, 'coord')
    edge_colors = [G[u][v].get('color', 'gray') for u, v in G.edges()]

    nx.draw_networkx_nodes(G, coord, ax=ax2, node_size=40, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, coord, ax=ax2, font_size=7)
    nx.draw_networkx_edges(G, coord, ax=ax2, edge_color=edge_colors, width=1.5)

    ax2.set_title(f"VRP Solution (Cost: {sol.cost:.2f})", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

# --- Execution ---

# Define the 3 instances you want to display
target_instances = ['A-n80-k10', 'B-n57-k9', 'M-n121-k7']

# Create a 3x2 figure
fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=100)

for idx, inst_name in enumerate(target_instances):
    # 1. Load your data (assuming your read/solve functions are available)
    vehcap = instances[inst_name]
    Node_list = read_vrp_instance(f"{inst_name}_input_nodes.txt")
    depot = Node_list[0]

    # 2. Perform Clustering (DBSCAN)
    coords = np.array([[n.x, n.y] for n in Node_list[1:]])
    db = DBSCAN(eps=15, min_samples=2).fit(coords)
    labels = db.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters: clusters[label] = []
        clusters[label].append(Node_list[i+1])

    # 3. Solve VRP
    sol = Solution()
    for label, cluster_nodes in clusters.items():
        cluster_routes = run_savings_on_cluster(cluster_nodes, depot, vehcap)
        for r in cluster_routes:
            sol.routes.append(r)
            sol.cost += r.cost

    # 4. Plot into the current row (idx)
    plot_instance_row(Node_list, clusters, sol, inst_name, axes[idx])

plt.tight_layout()
plt.show()
