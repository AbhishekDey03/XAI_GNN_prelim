import torch
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import add_self_loops, to_undirected
import umap

def threshold_adjacency(X, T=5, metric='hamming'):

    D = pairwise_distances(X, metric=metric)
    d_cut = np.percentile(D.reshape(-1), T)
    mask = (D < d_cut).astype(np.int8)
    mask = np.logical_or(mask, mask.T)  # Enforce symmetry
    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.float))
    
    return edge_index


def knn_adjacency(X, K=5, metric='jaccard'):
    n_nodes = X.shape[0]
    knn = NearestNeighbors(n_neighbors=K, metric=metric, n_jobs=-1).fit(X)
    _, knn_idx = knn.kneighbors(X)
    
    mask = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i, nbrs in enumerate(knn_idx):
        mask[i, nbrs] = 1

    mask = np.logical_or(mask, mask.T).astype(np.int8)  
    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.int64))
    
    edge_index = to_undirected(edge_index)             
    edge_index, _ = add_self_loops(edge_index)         

    return edge_index

def default_adjacency(X, original_edge_index):
    edge_index, _ = add_self_loops(original_edge_index)
    return edge_index


def UMAP_threshold(X, T=5, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    X_umap = reducer.fit_transform(X)
    D = pairwise_distances(X_umap)
    d_cut = np.percentile(D, T)
    mask = (D < d_cut).astype(int)
    mask = np.logical_or(mask, mask.T)
    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.float))
    return edge_index

def mknn_adjacency(X, K=5, metric='jaccard'):
    n_nodes = X.shape[0]
    knn = NearestNeighbors(n_neighbors=K, metric=metric, n_jobs=-1).fit(X)
    _, knn_idx = knn.kneighbors(X)

    mask = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i, nbrs in enumerate(knn_idx):
        mask[i, nbrs] = 1
    mask = np.logical_and(mask, mask.T).astype(np.int8)

    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.int64))
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    
    return edge_index