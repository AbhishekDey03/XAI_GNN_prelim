import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

import matplotlib.pyplot as plt
import numpy as np
from Silhouette_plots import plot_silhouette   


seed = 11363
torch.manual_seed(seed)
np.random.seed(seed)

type_map = {
    0: 'Theory',
    1: 'Reinforcement Learning',
    2: 'Genetic Algorithms',
    3: 'Neural Networks',
    4: 'Probabilistic Methods',
    5: 'Rule Learning',
    6: 'Case Based'
}

dataset = Planetoid(root='data/Planetoid', name='Cora')   
data = dataset[0]                                        

class FFNN(torch.nn.Module):
    """
    Simple fully-connected network:
        in → 256 → 128 → 64 → num_classes
    Dropout + ReLU between hidden layers.
    """
    def __init__(self, in_feats, out_feats, dropout=0.5):
        super().__init__()
        self.h1 = torch.nn.Linear(in_feats, 256)
        self.h2 = torch.nn.Linear(256, 128)
        self.h3 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, out_feats)
        self.do = torch.nn.Dropout(dropout)

    def forward(self, x, return_hidden=False):
        x = F.relu(self.h1(x))
        x = self.do(x)
        x = F.relu(self.h2(x))
        x = self.do(x)
        x = F.relu(self.h3(x))          
        hidden = x                     
        logits = self.out(x)
        return (logits, hidden) if return_hidden else logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FFNN(dataset.num_node_features, dataset.num_classes, dropout=0.5).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    logits = model(data.x)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x)
    pred = logits.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs  # [train, val, test]

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch {epoch:03d} │ Loss {loss:.4f} │ '
              f'Train/Val/Test {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}')

model.eval()
with torch.no_grad():
    _, H = model(data.x, return_hidden=True)    

H = H.cpu().numpy()
labels = data.y.cpu().numpy()

H_scaled = StandardScaler().fit_transform(H)

tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
emb_tsne = tsne.fit_transform(H_scaled)

umap_model = umap.UMAP(n_components=2, metric='euclidean', random_state=seed)
emb_umap = umap_model.fit_transform(H_scaled)




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sc1 = ax1.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels, cmap='Set1',
                  s=80, alpha=0.7)
ax1.set_title('t-SNE (FFNN, raw features)')
ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')

sc2 = ax2.scatter(emb_umap[:, 0], emb_umap[:, 1], c=labels, cmap='Set1',
                  s=80, alpha=0.7)
ax2.set_title('UMAP (FFNN, raw features)')
ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')

handles, lgnd_labels = sc1.legend_elements()
legend_names = [type_map[int(lbl)] for lbl in np.unique(labels)]
fig.legend(handles, legend_names, title='Paper Topic', loc='upper right', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('tsne_vs_umap_ffnn.pdf')
plt.show()
label_colors = sc1.cmap(sc1.norm(np.unique(labels)))
label_to_color = {label: col for label, col in zip(np.unique(labels), label_colors)}
label_names = {label: type_map[label] for label in np.unique(labels)}

plot_silhouette(data.x.cpu().numpy(), labels,
                'Silhouette — Original Features',
                label_to_color=label_to_color,
                label_names=label_names)

plot_silhouette(H, labels,
                'Silhouette — FFNN Final Hidden',
                save=True,
                plot_name='silhouette_ffnn_final_hidden.pdf',
                label_to_color=label_to_color,
                label_names=label_names)
