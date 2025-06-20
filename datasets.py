import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import re

seed = 11363
# Load and Setup the Cora datset, including labels and a graph visualization
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

g = to_networkx(data, to_undirected=True)

class_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}

unique_classes = np.unique(data.y.numpy())
cmap = plt.get_cmap('Set1', len(unique_classes))

fig, ax = plt.subplots(figsize=(9, 7))

pos = nx.spring_layout(g,seed = seed)

nx.draw(g,
        pos,
        ax=ax,
        with_labels=False,
        node_color=data.y,
        cmap=cmap,
        node_size=10,
        width=0.,
        edge_color='grey')

legend_handles = []
for i, class_id in enumerate(unique_classes):
    color = cmap(i)
    label = class_map[class_id]
    handle = plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=10,
                        label=label)
    legend_handles.append(handle)
ax.axis('on')

ax.legend(handles=legend_handles, title="Paper Topic", bbox_to_anchor=(1.05, 1))
ax.set_title('Cora Dataset Graph Visualization', size=16)
fig.subplots_adjust(right=0.7)

plt.show()

# tSNE embeddings of features
tsne = TSNE(n_components=2, random_state=seed)
features = data.x.numpy()
embeddings = tsne.fit_transform(features)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=data.y.numpy(), cmap='viridis', s=10)

handles, old_labels = scatter.legend_elements()

# Use regex to extract the integer from the formatted label string
new_labels = [class_map[int(re.search(r'\d+', label).group())] for label in old_labels]

ax.legend(handles, new_labels, title="Paper Topic")
ax.set_title('t-SNE Visualization of Cora Node Features', size=16)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
plt.savefig('tSNE_cora_node_features.png', bbox_inches='tight')
plt.show()