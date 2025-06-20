import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import pairwise_distances
import numpy as np
import torch_geometric.nn as pyg_nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import networkx as nx
from torch_geometric.utils import to_networkx
from Silhouette_plots import plot_silhouette


class_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}
do_train = True
plot_new_graph = False
seed = 11363
torch.manual_seed(seed);  np.random.seed(seed)

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
X = data.x.numpy()
n_nodes = X.shape[0]
metric = 'hamming' # 'jaccard' or 'hamming'
T = 5 

D = pairwise_distances(X, metric=metric)          

d_cut = np.percentile(D.reshape(-1), T)
mask  = (D < d_cut).astype(np.int8)

mask = np.logical_or(mask, mask.T) # Enforce symmetry

edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.float)) # Apply threshold to adjacency

print(f'Number of edges (incl. self-loops): {edge_index.shape[1]}')


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid, out_feats):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_feats, hid)
        self.conv2 = pyg_nn.GCNConv(hid, out_feats)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = self.conv2(x, ei)
        return x

new_data = data.clone()
new_data.edge_index = edge_index 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
data   = new_data.to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    opt.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward() 
    opt.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    logits = model(data).argmax(1)
    res = []
    for m in (data.train_mask, data.val_mask, data.test_mask):
        res.append((logits[m]==data.y[m]).float().mean().item())
    return res

if do_train:
    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            tr, va, te = test()
            print(f'E{epoch:03d} | loss {loss:.6f} | acc tr/va/te {tr:.6f}/{va:.6f}/{te:.6f}')

model.eval()
with torch.no_grad():
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index
    h1 = torch.relu(model.conv1(x, edge_index))
    h2 = torch.relu(model.conv2(h1, edge_index)) 


H = h2.cpu().numpy()
labels = data.y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=seed)
latent_embeddings = tsne.fit_transform(H)

fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1],
                     c=data.y.cpu().numpy(), cmap='Set1', s=80, alpha=0.7)

handles, old_labels = scatter.legend_elements()


new_labels = [class_map[int(re.search(r'\d+', label).group())] for label in old_labels]

ax.legend(handles, new_labels, title="Paper Topic")
ax.set_title('t-SNE of GCN Latent Node Embeddings, Threshold Adjacency', size=16)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
plt.savefig('tSNE_gcn_latent_embeddings_threshold.png', dpi=300)
plt.show()  


if plot_new_graph == True:
    G = to_networkx(new_data, to_undirected=True, remove_self_loops=True)
    pos = nx.spring_layout(G, seed=seed) 

    plt.figure(figsize=(10, 10))

    nx.draw_networkx_edges(G, pos, alpha=0.2)

    node_colors = data.y.cpu().numpy()
    nx.draw_networkx_nodes(
        G, pos,
        node_size=50,
        node_color=node_colors,
        cmap='Set1',
        alpha=0.8
    )

    plt.title('Graph Structure from Data-Driven Adjacency', size=16)
    plt.axis('off')
    plt.show()


plot_silhouette(X, labels, 'Silhouette — Original Input Features')
plot_silhouette(H, labels, 'Silhouette — GCN Final Layer Thresholded Adjacency',
                 save=True, plot_name='silhouette_gcn_final_layer_threshold.png')
