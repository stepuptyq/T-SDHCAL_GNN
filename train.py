import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloader import *
from utils import *
import numpy as np
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 64
num_epochs = 5
learning_rate = 1e-5
model_type = 1  # 1: fc with time; 2: fc with no time; 3: gnn with time

# Data Loading 
if model_type == 1:
    train_data = np.load("data/train_data.npz", allow_pickle=True)
    test_data = np.load("data/test_data.npz", allow_pickle=True)
elif model_type == 2:
    train_data = np.load("data/train_data_no_t.npz", allow_pickle=True)
    test_data = np.load("data/test_data_no_t.npz", allow_pickle=True)

# Rating Normalization
train_rating = train_data['R']
data_mean = train_rating.mean()
data_std = train_rating.std()
train_data_R = (train_data['R'] - data_mean) / data_std
test_data_R = (test_data['R'] - data_mean) / data_std
# print(f"train_data_R: {train_data_R}")
# print(f"test_data_R: {test_data_R}")

# Maximal Sequence Length
print("Calculating Maximal Sequence Length:")
train_data_I = train_data['I']
test_data_I = test_data['I']
max_seq_len = max(count_max_seq_len(train_data_I), count_max_seq_len(test_data_I))

print(f"Max Sequence Length: {max_seq_len}")

# Create Dataset instances for training and testing
if model_type == 1:
    train_dataset = CustomDataset_t(train_data, train_data_R, max_seq_len)
    test_dataset = CustomDataset_t(test_data, test_data_R, max_seq_len)
elif model_type == 2:
    train_dataset = CustomDataset(train_data, train_data_R, max_seq_len)
    test_dataset = CustomDataset(test_data, test_data_R, max_seq_len)

print(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("running on GPU")
else:
    print("running on CPU")

# Initialize model, criterion, and optimizer
from my_model import * # TransformerModel
# model = TransformerModel(max_input_length=max_seq_len).to(device)
if model_type == 1:
    model = FullyConnectedModel_t(max_input_length=max_seq_len).to(device)
elif model_type == 2:
    model = FullyConnectedModel(max_input_length=max_seq_len).to(device)
# model = GNNModel(max_input_length=max_seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

def data_processing(batch, device):
    if model_type == 1:
        input_indices1, input_indices2, input_indices3, input_indices4, masks, targets = batch
    elif model_type == 2:
        input_indices1, input_indices2, input_indices3, masks, targets = batch

    # Type Format & Move to GPU
    input_indices1 = input_indices1.long().to(device)
    input_indices2 = input_indices2.long().to(device)
    input_indices3 = input_indices3.long().to(device)
    if model_type == 1:
        input_indices4 = input_indices4.long().to(device)
    masks = masks.bool().to(device)
    targets = targets.float().to(device)
    if model_type == 1:
        return input_indices1, input_indices2, input_indices3, input_indices4, masks, targets
    elif model_type == 2:
        return input_indices1, input_indices2, input_indices3, masks, targets

def train(batch, model, device, criterion, optimizer):
    # set the model in train mode
    model.train()
    if model_type == 1:
        input_indices1, input_indices2, input_indices3, input_indices4, masks, targets = data_processing(batch, device)
        # print('input_indices1:')
        # print(input_indices1)
        optimizer.zero_grad()
        outputs = model(input_indices1, input_indices2, input_indices3, input_indices4, masks, device).squeeze()
    elif model_type == 2:
        input_indices1, input_indices2, input_indices3, masks, targets = data_processing(batch, device)
        optimizer.zero_grad()
        outputs = model(input_indices1, input_indices2, input_indices3, masks, device).squeeze()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss

def test(batch, model, device, criterion):
    if model_type == 1:
        input_indices1, input_indices2, input_indices3, input_indices4, masks, targets = data_processing(batch, device)
        test_outputs = model(input_indices1, input_indices2, input_indices3, input_indices4, masks, device).squeeze()
    elif model_type == 2:
        input_indices1, input_indices2, input_indices3, masks, targets = data_processing(batch, device)
        test_outputs = model(input_indices1, input_indices2, input_indices3, masks, device).squeeze()
    test_loss = criterion(test_outputs, targets)
    return test_loss

def full_eval(test_loader, model, device, criterion):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            test_loss = test(batch, model, device, criterion)
            total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    return avg_test_loss

# Training loop with DataLoader
for epoch in range(num_epochs):
    # Model training

    total_train_loss = 0

    for batch in tqdm(train_loader):
        # step += 1
        # Batch: 6 tensors of size (64, 1033)
        loss = train(batch, model, device, criterion, optimizer)
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
        
    scheduler.step()
    avg_test_loss = full_eval(test_loader, model, device, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

   