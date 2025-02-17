import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_GNN import *
from utils import *
import numpy as np
from tqdm import tqdm
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 64
num_epochs = 5
learning_rate = 1e-5

# Load Dataset instances for training and testing
print('Loading Dataset ...')
# train_dataset = torch.load('dataset_train.pt', weights_only=False)
# test_dataset = torch.load('dataset_test.pt', weights_only=False)
train_dataset = torch.load('dataset_train_directed.pt', weights_only=False)
test_dataset = torch.load('dataset_test_directed.pt', weights_only=False)
print('Dataset Loaded!')

# train_dataset = [Data(x=[549, 4], edge_index=[2, 21266], y=30.0),
#                  Data(x=[511, 4], edge_index=[2, 13218], y=30.0),
#                  Data(x=[213, 4], edge_index=[2, 2742], y=10.0),
#                  ...]

# Normalization

print('Normalizing Dataset ...')
train_rating = np.zeros(len(train_dataset))
test_rating = np.zeros(len(test_dataset))

for i in range(len(train_dataset)):
    train_rating[i] = train_dataset[i].y
for i in range(len(test_dataset)):
    test_rating[i] = test_dataset[i].y

data_mean = train_rating.mean()
data_std = train_rating.std()
train_data_R = (train_rating - data_mean) / data_std
test_data_R = (test_rating - data_mean) / data_std

for i, data in enumerate(train_dataset):
    data.y = torch.tensor([train_data_R[i]])
for i, data in enumerate(test_dataset):
    data.y = torch.tensor([test_data_R[i]])
print('Normalization Finished!')

print('Loading Data ...')
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print('Data Loaded!')
print(type(train_loader))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("running on GPU")
else:
    print("running on CPU")

# Initialize model, criterion, and optimizer
from my_model import *
# model = DGCNN().to(device)
model = GNNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

def train(batch, model, device, criterion, optimizer):
    # set the model in train mode
    model.train()
    optimizer.zero_grad()
    outputs = model(batch)
    loss = criterion(outputs, batch.y.float().to(device))
    loss.backward()
    optimizer.step()
    return loss

def test(batch, model, device, criterion):
    test_outputs = model(batch)
    test_loss = criterion(test_outputs, batch.y.float().to(device))
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
        loss = train(batch, model, device, criterion, optimizer)
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
        
    scheduler.step()
    avg_test_loss = full_eval(test_loader, model, device, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

   