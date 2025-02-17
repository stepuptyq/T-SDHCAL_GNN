from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm

dist_threshold = 10
time_threshold = 0.1

def create_pads(input_array, target_length, padding_value):
    # Pad the array to the target length with the specified padding value
    padded_array = np.full(target_length, padding_value, dtype=input_array.dtype)
    padded_array[:len(input_array)] = input_array
    
    return padded_array #, padding_mask # , attention_mask

def batch_create_pads(input_arrays, target_length, padding_value=0):
    batch_input_with_pads = np.zeros((len(input_arrays), target_length))
    for i, a in enumerate(input_arrays):
        # 0 denotes padding token, so add each value by 1
        a += 1
        padded_a = create_pads(a, target_length, padding_value)
        # padded_a += 1
        batch_input_with_pads[i,:] = padded_a
        
    return batch_input_with_pads # , masks

def create_masks(input_array_len, target_length):
    # # Generate padding mask (1 for real tokens and 0 for padding)
    padding_mask = np.zeros(target_length).astype(int)
    padding_mask[:input_array_len] = 1
    return padding_mask

def batch_create_masks(input_arrays, target_length):
    masks = np.zeros((len(input_arrays), target_length)).astype(int)
    for i, a in enumerate(input_arrays):
        masks[i,:] = create_masks(len(a), target_length)
    return masks

from torch_geometric.data import Data

# For each instance
def build_custom_edge_index(data, dist_threshold, time_threshold):
    num_nodes = len(data)
    edge_index = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                # Calculate the Euclidean distance and time difference
                distance = np.linalg.norm(data[i][0:3] - data[j][0:3])
                time_diff = abs(data[i][3] - data[j][3])
                if distance < dist_threshold and time_diff < time_threshold:
                    edge_index.append([i, j])

    return torch.tensor(edge_index).T

def Data_loader_GNN(data, data_R):
    d_length = len(data)
    dataset = []
    for i in range (d_length):
        node_features1 = torch.tensor(data['data'][i], dtype=torch.float)
        edge_index1 = build_custom_edge_index(data['data'][i], dist_threshold, time_threshold)
        y1 = data_R[i] # Particle energy R
        dataset.append(Data(x=node_features1, edge_index=edge_index1, y=y1))
    return dataset

class CustomDataset_GNN(Dataset):
    def __init__(self, data, rating, max_seq_len):
        self.targets = rating
        # print("Creating padded I ...")
        self.indices1 = batch_create_pads(data['data'], max_seq_len, padding_value=0)
        # print("Creating padded J ...")
        self.indices2 = batch_create_pads(data['data'], max_seq_len, padding_value=0)
        # print("Creating padded K ...")
        self.indices3 = batch_create_pads(data['data'], max_seq_len, padding_value=0)
        # print("Creating padded T ...")
        self.indices4 = batch_create_pads(data['T'], max_seq_len, padding_value=0)

        self.masks = batch_create_masks(data['I'], max_seq_len)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.indices1[idx], self.indices2[idx], self.indices3[idx], self.indices4[idx], self.masks[idx], self.targets[idx])

