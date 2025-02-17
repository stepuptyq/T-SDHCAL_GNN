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
    使用 PyTorch 张量操作在 GPU 上构建 edge_index。
    """
    # 确保输入数据为 PyTorch 张量，并移动到 GPU
    data = torch.tensor(data, dtype=torch.float, device=device)

    # 提取位置 (x, y, z) 和时间 t
    positions = data[:, :3]  # 位置 (x, y, z)
    times = data[:, 3]       # 时间 t

    # 计算所有节点对的欧几里得距离和时间差
    num_nodes = data.size(0)
    positions_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # 广播计算位置差
    distances = torch.norm(positions_diff, dim=-1)  # 计算欧几里得距离

    times_diff = torch.abs(times.unsqueeze(1) - times.unsqueeze(0))  # 广播计算时间差

    # 找到满足条件的边（距离和时间差均小于阈值）
    mask = (distances < dist_threshold) & (times_diff < time_threshold) & (torch.arange(num_nodes, device=device).unsqueeze(1) != torch.arange(num_nodes, device=device).unsqueeze(0))

    # 获取边的索引
    edge_index = torch.nonzero(mask, as_tuple=False).T  # 转置为 [2, num_edges] 格式
    return edge_index

def build_custom_edge_time(data, time_threshold, device='cpu'):
    """
    构建边索引矩阵，只基于时间差进行计算。
    
    参数:
        - data: 包含时间戳的节点特征矩阵，形状为 (N, d)，其中第 d 列为时间戳。
        - time_threshold: 时间差阈值，单位与时间戳一致。
        - device: 指定设备（'cpu' 或 'cuda'）。
    
    返回:
        - edge_index: 边索引矩阵，形状为 (2, num_edges)。
    """
    # 假设时间戳是节点特征矩阵的最后一列
    timestamps = data[:, -1]  # 提取时间戳列，形状为 (N,)
    num_nodes = timestamps.shape[0]
    
    # 初始化空的边索引列表
    edge_list = []

    # 遍历每对节点，基于时间差构建边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 排除自环
                time_diff = abs(timestamps[i] - timestamps[j])  # 计算时间差
                if time_diff <= time_threshold:  # 满足时间差条件
                    edge_list.append([i, j])  # 添加边 (i, j)

    # 转换为 PyTorch 张量并移动到指定设备
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return edge_index

def build_custom_edge_time_directed(data, time_threshold, device='cpu'):
    """
    构建边索引矩阵，只基于时间差进行计算。
    
    参数:
        - data: 包含时间戳的节点特征矩阵，形状为 (N, d)，其中第 d 列为时间戳。
        - time_threshold: 时间差阈值，单位与时间戳一致。
        - device: 指定设备（'cpu' 或 'cuda'）。
    
    返回:
        - edge_index: 边索引矩阵，形状为 (2, num_edges)。
    """
    # 假设时间戳是节点特征矩阵的最后一列
    timestamps = data[:, -1]  # 提取时间戳列，形状为 (N,)
    num_nodes = timestamps.shape[0]
    
    # 初始化空的边索引列表
    edge_list = []

    # 遍历每对节点，基于时间差构建边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and timestamps[j] >= timestamps[i]:  # 排除自环，根据时间计算有向图
                time_diff = abs(timestamps[i] - timestamps[j])  # 计算时间差
                if time_diff <= time_threshold:  # 满足时间差条件
                    edge_list.append([i, j])  # 添加边 (i, j)

    # 转换为 PyTorch 张量并移动到指定设备
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return edge_index
    
def Data_loader_GNN(data, device='cuda'):
    """
    加载数据并构建 PyTorch Geometric 的 Dataset。
    """
    d_length = len(data['data'])
    dataset = []
    for i in tqdm(range(d_length)):
        # 将节点特征移动到 GPU
        node_features1 = torch.tensor(data['data'][i], dtype=torch.float, device=device)
        # 使用 GPU 加速构建 edge_index
        # edge_index1 = build_custom_edge_index(data['data'][i], dist_threshold, time_threshold, device=device)
        edge_index1 = build_custom_edge_time_directed(data['data'][i], time_threshold, device=device)
        # edge_index1 = build_custom_edge_time(data['data'][i], time_threshold)
        # 获取目标值并移动到 GPU
        y1 = torch.tensor(data['R'][i], dtype=torch.float, device=device)  # 粒子能量 R
        # 构建 PyTorch Geometric 的 Data 对象
        dataset.append(Data(x=node_features1, edge_index=edge_index1, y=y1))
    return dataset

# 检查设备
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