import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import re
import networkx as nx
from torch_geometric.utils import to_networkx
from Silhouette_plots import plot_silhouette
from Adjacency_types import threshold_adjacency, knn_adjacency, default_adjacency
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler



type_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}

adjacency_type = 'KNN'  # 'default', 'threshold', 'knn'
blank  = '' # Placeholder for blanks in the plots
seed = 11363

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
X = data.x.numpy()

if adjacency_type.lower() == 'threshold':
    metric = 'Jaccard'
    T = 5
    edge_index = threshold_adjacency(X, T, metric.lower())
    print(f'Number of edges (incl. self-loops): {edge_index.shape[1]}')
elif adjacency_type.lower() == 'knn':
    K = 5
    metric = 'Jaccard'
    edge_index = knn_adjacency(X, K, metric.lower())
    print(f'Number of edges (incl. self-loops): {edge_index.shape[1]}')
elif adjacency_type.lower() == 'default':
    metric = False
    edge_index = default_adjacency(X, data.edge_index)
    print(f'Number of edges (incl. self-loops): {edge_index.shape[1]}')

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid, out_feats):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_feats, hid)
        self.conv2 = pyg_nn.GCNConv(hid, hid) 
        self.conv3 = pyg_nn.GCNConv(hid, out_feats)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = self.conv3(x, ei)                     
        return x    
    

new_data = data.clone()
new_data.edge_index = edge_index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
data = new_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

torch.no_grad()
def test():
    model.eval()
    pred = model(data).argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs  # [train_acc, val_acc, test_acc]


for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test()
            print(f'Epoch {epoch:03d} | Loss: {loss:.6f} | Train/Val/Test Acc: {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}')

model.eval()
with torch.no_grad():
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index
    h1 = torch.relu(model.conv1(x, edge_index))
    h2 = torch.relu(model.conv2(h1, edge_index))

# t-SNE visualization
H = h2.cpu().numpy()
labels = data.y.cpu().numpy()
# Default umap and TSNE can look bad, try preprocessing
H_scaled = StandardScaler().fit_transform(H)


tsne = TSNE(n_components=2, random_state=seed,perplexity=30)
embeds_2d = tsne.fit_transform(H_scaled)
labels = data.y.cpu().numpy()
# UMAP visualization
umap_model = umap.UMAP(
    n_components=2,
    metric='euclidean',
    random_state=seed
)
embeds_umap = umap_model.fit_transform(H_scaled)

# Plot both to compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

scatter_tsne = ax1.scatter(embeds_2d[:, 0],embeds_2d[:, 1],c=labels,cmap='Set1',s=80,alpha=0.7)
ax1.set_title(f't-SNE — {adjacency_type} {metric if metric else blank}', size=14)
ax1.set_xlabel('t-SNE Dim 1')
ax1.set_ylabel('t-SNE Dim 2')

scatter_umap = ax2.scatter(
    embeds_umap[:, 0],
    embeds_umap[:, 1],
    c=labels,
    cmap='Set1',
    s=80,
    alpha=0.7
)
ax2.set_title(f'UMAP — {adjacency_type} {metric if metric else blank}', size=14)
ax2.set_xlabel('UMAP Dim 1')
ax2.set_ylabel('UMAP Dim 2')

handles, legend_labels = scatter_tsne.legend_elements()
legend_names = [type_map[int(re.search(r'\d+', lbl).group())] for lbl in legend_labels]
fig.legend(handles, legend_names, title="Paper Topic", loc='upper right', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.95, 1])  # leaves space for legend
plt.savefig(f'tSNE_vs_UMAP_{adjacency_type}{metric if metric else blank}.png')
plt.show()




# Get silhouette labels
scatter = ax1.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap='Set1', s=80, alpha=0.7)
handles, legend_labels = scatter.legend_elements()
colors = scatter.cmap(scatter.norm(np.unique(labels)))
label_to_color = {label: color for label, color in zip(np.unique(labels), colors)}
label_names = {label: type_map[label] for label in np.unique(labels)}

plot_silhouette(X, labels, 'Silhouette — Original Input Features',
                label_to_color=label_to_color, label_names=label_names)

plot_silhouette(H, labels, f'Silhouette — GCN Final Layer {adjacency_type} {metric if metric else blank}',
                save=True,
                plot_name=f'silhouette_gcn_final_layer_{adjacency_type}{metric if metric else blank}.png',
                label_to_color=label_to_color,
                label_names=label_names)

