from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class CustomDataset(Dataset):
    def __init__(self, data, rating, max_seq_len):
        self.targets = rating
        # print("Creating padded I ...")
        self.indices1 = batch_create_pads(data['I'], max_seq_len, padding_value=0)
        # print("Creating padded J ...")
        self.indices2 = batch_create_pads(data['J'], max_seq_len, padding_value=0)
        # print("Creating padded K ...")
        self.indices3 = batch_create_pads(data['K'], max_seq_len, padding_value=0)

        self.masks = batch_create_masks(data['I'], max_seq_len)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.indices1[idx], self.indices2[idx], self.indices3[idx], self.masks[idx], self.targets[idx])

class CustomDataset_t(Dataset):
    def __init__(self, data, rating, max_seq_len):
        self.targets = rating
        # print("Creating padded I ...")
        self.indices1 = batch_create_pads(data['I'], max_seq_len, padding_value=0)
        # print("Creating padded J ...")
        self.indices2 = batch_create_pads(data['J'], max_seq_len, padding_value=0)
        # print("Creating padded K ...")
        self.indices3 = batch_create_pads(data['K'], max_seq_len, padding_value=0)
        # print("Creating padded K ...")
        self.indices4 = batch_create_pads(data['T'], max_seq_len, padding_value=0)

        self.masks = batch_create_masks(data['I'], max_seq_len)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.indices1[idx], self.indices2[idx], self.indices3[idx], self.indices4[idx], self.masks[idx], self.targets[idx])

