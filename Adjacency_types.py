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

def threshold_adjacency(X, T=5, metric='hamming'):
    """
    Create an adjacency matrix based on pairwise distances with a threshold.
    
    Parameters:
    - X: Node features (numpy array).
    - T: Threshold percentile for edge creation.
    - metric: Distance metric to use ('jaccard' or 'hamming').
    
    Returns:
    - edge_index: Edge index tensor for the graph.
    """
    D = pairwise_distances(X, metric=metric)
    d_cut = np.percentile(D.reshape(-1), T)
    mask = (D < d_cut).astype(np.int8)
    mask = np.logical_or(mask, mask.T)  # Enforce symmetry
    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.float))
    
    return edge_index

def knn_adjacency(X, K=5, metric='jaccard'):
    """
    Create an adjacency matrix based on K-nearest neighbors.
    
    Parameters:
    - X: Node features (numpy array).
    - K: Number of nearest neighbors.
    - metric: Distance metric to use ('jaccard' or 'hamming').
    
    Returns:
    - edge_index: Edge index tensor for the graph.
    """
    
    n_nodes = X.shape[0]
    knn = NearestNeighbors(n_neighbors=K, metric=metric, n_jobs=-1).fit(X)
    _, knn_idx = knn.kneighbors(X)
    
    mask = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i, nbrs in enumerate(knn_idx):
        mask[i, nbrs] = 1
    
    mask = np.logical_or(mask, mask.T).astype(np.int8)
    edge_index, _ = dense_to_sparse(torch.tensor(mask, dtype=torch.int64))
    
    return edge_index

def default_adjacency(X):
    """
    Create a default adjacency matrix with self-loops.
    
    Parameters:
    - X: Node features (numpy array).
    
    Returns:
    - edge_index: Edge index tensor for the graph.
    """
    n_nodes = X.shape[0]
    edge_index = torch.tensor([[i for i in range(n_nodes)], [i for i in range(n_nodes)]], dtype=torch.long)
    
    return edge_index
