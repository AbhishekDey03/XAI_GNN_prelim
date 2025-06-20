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
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import add_self_loops, to_undirected


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

def default_adjacency(X, original_edge_index=None):
    if original_edge_index is None:
        raise ValueError("Must provide original edge index from dataset.")
    edge_index, _ = add_self_loops(original_edge_index)
    return edge_index
