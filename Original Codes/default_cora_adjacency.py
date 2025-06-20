import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import networkx as nx
from torch_geometric.utils import to_networkx
from Silhouette_plots import plot_silhouette



type_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}

do_train = True
plot_graph = False
seed = 11363

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
X = data.x.numpy()


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_feats, hid_feats)
        self.conv2 = pyg_nn.GCNConv(hid_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

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

if do_train:
    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test()
            print(f'Epoch {epoch:03d} | Loss: {loss:.6f} | Train/Val/Test Acc: {train_acc:.6f}/{val_acc:.6f}/{test_acc:.6f}')

model.eval()
with torch.no_grad():
    edge_index = data.edge_index
    h1 = F.relu(model.conv1(data.x, edge_index))
    h2 = model.conv2(h1, edge_index)

# t-SNE visualization
H = h2.cpu().numpy()
labels = data.y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
embeds_2d = tsne.fit_transform(H)
labels = data.y.cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap='Set1', s=80, alpha=0.7)
handles, legend_labels = scatter.legend_elements()
legend_names = [type_map[int(re.search(r'\d+', lbl).group())] for lbl in legend_labels]
ax.legend(handles, legend_names, title="Paper Topic")
ax.set_title('t-SNE of GCN Latent Embeddings (Default Cora Graph)', size=16)
ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2')
plt.savefig('tSNE_cora_default_graph_GCN_hidden2.png', bbox_inches='tight')
plt.show()

if plot_graph:
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=labels, cmap='Set1', alpha=0.8)
    plt.title('Default Cora Citation Graph Structure', size=16)
    plt.axis('off')
    plt.show()


plot_silhouette(X, labels, 'Silhouette — Original Input Features')
plot_silhouette(H, labels, 'Silhouette — GCN Final Layer', plot_name='silhouette_gcn_final_layer.png', save=True)
