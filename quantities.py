import numpy as np
import torch
import networkx as nx

from sklearn.metrics import pairwise_distances, silhouette_score
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops




def modularity(edge_index,labels,num_nodes):
    edge_index, _ = remove_self_loops(edge_index)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    communities = [set(np.where(labels == c)[0]) for c in np.unique(labels)]
    return nx.algorithms.community.modularity(g, communities)


def n_components(edge_index,num_nodes):
    temp_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(temp_data, to_undirected=True)
    return nx.number_connected_components(g)

def within_edge_ratio(edge_index,labels):
    src, dst = edge_index.cpu().numpy()
    mask = src != dst 
    same = labels[src] == labels[dst]
    return same[mask].mean()


def intra_inter_edge_distance_ratio(X,edge_index,labels,metric= 'jaccard'):
    src, dst = edge_index.cpu().numpy()
    mask = src != dst 
    src, dst = src[mask], dst[mask]

    D = pairwise_distances(X, metric=metric)
    edge_d = D[src, dst]

    same_class = labels[src] == labels[dst]
    intra = edge_d[same_class]
    inter = edge_d[~same_class]

    if len(intra) == 0 or len(inter) == 0:
        return np.nan

    return intra.mean() / inter.mean()


def average_silhouette(embeddings,labels, metric = 'euclidean'):
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(embeddings, labels, metric=metric)


def graph_quantities(edge_index,features,embeddings,labels,metric_pairwise = 'jaccard'):
    """
    Computes all core quantities used for structural and embedding analysis:
        - M: Modularity
        - C: Number of Connected Components
        - W: Within-class edge ratio
        - R: Intra/Inter edge distance ratio
        - S: Silhouette Score on Embeddings
    """
    num_nodes = features.shape[0]

    return {
        'M': modularity(edge_index, labels, num_nodes),
        'C': n_components(edge_index, num_nodes),
        'W': within_edge_ratio(edge_index, labels),
        'R': intra_inter_edge_distance_ratio(features, edge_index, labels, metric_pairwise),
        'S': average_silhouette(embeddings, labels)
    }