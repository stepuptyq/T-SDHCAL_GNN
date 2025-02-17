import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloader_GNN import *
from utils import *
import numpy as np
from tqdm import tqdm
import os
from torch_geometric.data import Data

dist_threshold = 10
time_threshold = 0.1

def build_custom_edge_index(data, dist_threshold, time_threshold, device='cuda'):
    """
    Use PyTorch tensor operations on GPU to build edge_index。
    """
    # Ensure the input data is a PyTorch tensor and moved to GPU
    data = torch.tensor(data, dtype=torch.float, device=device)

    # Extract positions (x, y, z) and time
    positions = data[:, :3]  # pos (x, y, z)
    times = data[:, 3]       # time t

    # Calculate the Euclidean distance and time difference between all pairs of nodes
    num_nodes = data.size(0)
    positions_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # Broadcast calculation of position differences
    distances = torch.norm(positions_diff, dim=-1)  # Calculate the Euclidean distance

    times_diff = torch.abs(times.unsqueeze(1) - times.unsqueeze(0))  # Broadcast calculation of time differences

    # Find edges that meet the conditions (both distance and time difference are less than the threshold).
    mask = (distances < dist_threshold) & (times_diff < time_threshold) & (torch.arange(num_nodes, device=device).unsqueeze(1) != torch.arange(num_nodes, device=device).unsqueeze(0))

    # Get the index of the edge
    edge_index = torch.nonzero(mask, as_tuple=False).T  # Transpose to the format [2, num_edges]
    return edge_index

def build_custom_edge_time(data, time_threshold, device='cpu'):
    """
    Construct the edge index matrix based solely on time differences.

    Parameters:
        - data: A node feature matrix containing timestamps, with shape (N, d), where the d-th column is the timestamp.
        - time_threshold: The time difference threshold, in the same unit as the timestamps.
        - device: Specifies the device ('cpu' or 'cuda').

    Returns:
        - edge_index: The edge index matrix, with shape (2, num_edges).
    """
    # Assume that the timestamps are in the last column of the node feature matrix
    timestamps = data[:, -1]  # Extract the timestamp column, with shape (N,)
    num_nodes = timestamps.shape[0]
    
    # Initialize an empty edge index list
    edge_list = []

    # Iterate through each pair of nodes to construct edges based on time differences
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                time_diff = abs(timestamps[i] - timestamps[j])  # Calculate the time difference
                if time_diff <= time_threshold:  # Satisfy the time difference condition
                    edge_list.append([i, j])  # Add edge (i, j)

    # Convert to a PyTorch tensor and move to the specified device
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return edge_index

def build_custom_edge_time_directed(data, time_threshold, device='cpu'):
    timestamps = data[:, -1]
    num_nodes = timestamps.shape[0]
    
    edge_list = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and timestamps[j] >= timestamps[i]:  # Exclude self-loops and construct a directed graph based on time
                time_diff = abs(timestamps[i] - timestamps[j])
                if time_diff <= time_threshold:
                    edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return edge_index
    
def Data_loader_GNN(data, device='cuda'):
    """
    Load data and construct a Dataset for PyTorch Geometric
    """
    d_length = len(data['data'])
    dataset = []
    for i in tqdm(range(d_length)):
        # Move node features to the GPU
        node_features1 = torch.tensor(data['data'][i], dtype=torch.float, device=device)
        # Use GPU acceleration to construct edge_index
        # edge_index1 = build_custom_edge_index(data['data'][i], dist_threshold, time_threshold, device=device)
        edge_index1 = build_custom_edge_time_directed(data['data'][i], time_threshold, device=device)
        # edge_index1 = build_custom_edge_time(data['data'][i], time_threshold)
        # Get the target values and move them to the GPU
        y1 = torch.tensor(data['R'][i], dtype=torch.float, device=device)  # Particle energy R
        # Construct a Data object for PyTorch Geometric
        dataset.append(Data(x=node_features1, edge_index=edge_index1, y=y1))
    return dataset

# Check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("running on GPU")
else:
    print("running on CPU")

train_data = np.load("data/train_data_GNN.npz", allow_pickle=True)
dataset = Data_loader_GNN(train_data, device=device)
torch.save(dataset, 'dataset_train_directed.pt')
test_data = np.load("data/test_data_GNN.npz", allow_pickle=True)
dataset = Data_loader_GNN(test_data, device=device)
torch.save(dataset, 'dataset_test_directed.pt')