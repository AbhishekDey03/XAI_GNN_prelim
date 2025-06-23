import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Adjacency_types as adj

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

cmap = plt.get_cmap('Set1', len(unique_classes))
node_colors = [cmap(label) for label in labels]

overlap_num = 200
KNN_num = 5
mKNN_num = 15
threshold_percentile = 5
adj_methods = {
    'Threshold\n(Jaccard, T=5%)': lambda X: adj.threshold_adjacency(X, T=threshold_percentile, metric='jaccard'),
    'kNN\n(Jaccard, K=5)': lambda X: adj.knn_adjacency(X, K=KNN_num, metric='jaccard'),
    'Default\n(Original Graph)': lambda X: adj.default_adjacency(X, data.edge_index),
    'mKNN\n(Jaccard, K=15)': lambda X: adj.mknn_adjacency(X, K=mKNN_num, metric='jaccard'),
    f'Raw Overlap\n(min {overlap_num} shared features)': lambda X: adj.raw_overlap_adjacency(X, min_overlap=overlap_num),
}


n_rows = 2

n_methods = len(adj_methods)
n_cols = np.ceil(n_methods / n_rows)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows), squeeze=False)
axes = axes.flatten()

for ax, (title, build_fn) in zip(axes, adj_methods.items()):
    edge_index = build_fn(X)
    temp_data = Data(x=data.x, edge_index=edge_index, y=data.y)
    g = to_networkx(temp_data, to_undirected=True)
    g.remove_edges_from(nx.selfloop_edges(g))
    pos = nx.spring_layout(g, seed=seed)

    nx.draw(
        g,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        with_labels=False,
        node_size=12,
        edge_color='lightgray',
        alpha=0.8
    )

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

# Turn off unused axes if any
for ax in axes[n_methods:]:
    ax.axis('off')

# Legend
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(i), markersize=9,
               label=class_map[i])
    for i in unique_classes
]

fig.legend(handles=legend_handles, title="Paper Topic",
           bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=10, title_fontsize=11)

fig.suptitle('Cora Dataset — Graphs from Different Adjacency Constructions',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('all_graph_adjacency_views.png', bbox_inches='tight', dpi=300)
plt.show()