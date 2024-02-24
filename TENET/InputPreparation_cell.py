import hnswlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
import random
# This is just the demostration of the construction of the proximal interaction cells and inter-mediate interaction cells to the  query cell.
data = pd.read_csv('coord.csv')

# HNSW INDEX
dim = data.shape[1]
num_elements = data.shape[0]

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=16)
p.set_ef(50)

p.add_items(data)

# Index for proximal interaction cells
# Define the number of the neighbor
k = 3
labels, distances = p.knn_query(data, k=k)

# Randomly choice of the query cell -> define to be ree
random_point = 1000

G_knn = nx.Graph()

# Add a node with node coordinates as node attributes
for i, coordinate in enumerate(data.values):
    G_knn.add_node(i, pos=tuple(coordinate))
# Cell spatial location
pos_ = nx.get_node_attributes(G_knn, 'pos')
nx.draw_networkx_nodes(G_knn, pos_, node_color='purple', node_size=3)

# Add edges
for i, neighbors in enumerate(labels):
    for neighbor in neighbors:
        if neighbor != i:
            G_knn.add_edge(i, neighbor)

# Result for the spatial connectivity network and the k proximal interaction cells
nx.draw_networkx_edges(G_knn, pos_, edge_color='black', width=0.4, style='-')
node_colors_knn = ['red' if i == random_point else 'green' if i in labels[random_point] else 'purple' for i in range(num_elements)]
node_size = [10 if i in labels[random_point] or i == random_point else 3 for i in range(num_elements)]
nx.draw_networkx_nodes(G_knn, pos_, node_color=node_colors_knn, node_size=node_size)
# plt.savefig('./i_alter_alter.png',format = 'png',dpi=300)
plt.show()
#==============================================================================================================================================
# Community discovery using Louvain algorithm
# Step 2 the lovain algorithm for cluster  coalesce
partition = community.best_partition(G_knn)
# the community id of the query cell
random_point_community = partition[1000]
for node in G_knn.nodes():
    if partition[node] == random_point_community:
        if node != random_point:
            G_knn.add_edge(random_point, node)

pos_community = nx.get_node_attributes(G_knn, 'pos')
node_colors_community = ['red' if node == random_point else 'green' if partition[node] == random_point_community else 'purple' for node in G_knn.nodes()]
node_sizes_community = [15 if partition[node] == random_point_community else 3 for node in G_knn.nodes()]
nx.draw_networkx_nodes(G_knn, pos_community, node_color=node_colors_community, node_size=node_sizes_community, alpha=0.8)
nx.draw_networkx_edges(G_knn, pos_community, edge_color='black', width=0.4, style='-')
# plt.savefig('./2_alter_alter.png',format = 'png',dpi=300)
plt.show()

# Another procedure can be added for dectecting the interaction cells in step1 and step2 -> if the interaction cells are not in step2, added to the graph