import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Adjacency_types as adj

# --- Setup ---
seed = 11363
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
X = data.x.numpy()
labels = data.y.numpy()
unique_classes = np.unique(labels)

class_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}

# Color map
cmap = plt.get_cmap('Set1', len(unique_classes))
node_colors = [cmap(label) for label in labels]

# --- Adjacency Methods ---
adj_methods = {
    'Threshold (Jaccard)': lambda X: adj.threshold_adjacency(X, T=5, metric='jaccard'),
    'kNN (Jaccard)': lambda X: adj.knn_adjacency(X, K=5, metric='jaccard'),
    'Default (Original)': lambda X: adj.default_adjacency(X, data.edge_index),
    'mKNN (Jaccard)': lambda X: adj.mknn_adjacency(X, K=10, metric='jaccard'),
    'Raw Overlap': lambda X: adj.raw_overlap_adjacency(X, min_overlap=300),
}

n_methods = len(adj_methods)
fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))

if n_methods == 1:
    axes = [axes] 

for ax, (name, build_fn) in zip(axes, adj_methods.items()):
    edge_index = build_fn(X)
    temp_data = Data(x=data.x, edge_index=edge_index, y=data.y)
    g = to_networkx(temp_data, to_undirected=True)
    pos = nx.spring_layout(g, seed=seed)
    g.remove_edges_from(nx.selfloop_edges(g))
    nx.draw(
        g,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        with_labels=False,
        node_size=10,
        edge_color='gray',
        alpha=0.8
    )
    ax.set_title(name, fontsize=12)
    ax.axis('off')

legend_handles = []
for i in unique_classes:
    handle = plt.Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor=cmap(i), markersize=10,
        label=class_map[i]
    )
    legend_handles.append(handle)

fig.legend(handles=legend_handles, title="Paper Topic", bbox_to_anchor=(1.02, 0.5), loc="center left")
fig.suptitle('Cora Dataset — Graphs from Different Adjacency Constructions', fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('all_graph_adjacency_views.png', bbox_inches='tight')
plt.show()