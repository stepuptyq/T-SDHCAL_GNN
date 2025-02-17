import numpy as np

# reading npz file：
def read_npz(file):
    loaded_data = np.load(f'data/pion{i * 10}Gev_GNN.npz', allow_pickle=True)
    load_data = {'data': loaded_data['data']}
    return load_data
    
concatenated_data = {}
for i in range(1,6):
    file = read_npz(f'data/pion{i * 10}Gev_GNN.npz')
    file['R'] = (np.ones(len(file['data'])) * 10 * i)
#     concatenated_data.update(file)
    for key in file.keys():
        if key in concatenated_data:
            concatenated_data[key]= np.concatenate((concatenated_data[key], file[key]), axis=0)
        else:
            concatenated_data[key]=file[key]

from sklearn.model_selection import train_test_split
import numpy as np

# Example data array of a specified size, let's say 100 samples
data_size = len(concatenated_data['data'])
data = np.arange(data_size)

# Split the dataset into training and testing sets
train_indices, test_indices = train_test_split(data, test_size=0.2, random_state=42)

import pandas as pd
# Convert indices to DataFrame
train_df = pd.DataFrame(train_indices, columns=['Index'])
test_df = pd.DataFrame(test_indices, columns=['Index'])

# Save the DataFrames to CSV files
train_df.to_csv('data/train_indices_GNN.csv', index=False)
test_df.to_csv('data/test_indices_GNN.csv', index=False)

print(len(concatenated_data['R']))

train_data = concatenated_data['data'][train_indices]
train_R = concatenated_data['R'][train_indices]

test_data = concatenated_data['data'][test_indices]
test_R = concatenated_data['R'][test_indices]

# Save train and test I, J, K with npz format
np.savez('data/train_data_GNN.npz', data=train_data, R=train_R)
np.savez('data/test_data_GNN.npz', data=test_data, R=test_R)


# # Find the max lenght for each element in I
# I = concatenated_data['I']
# max_length = 0
# for i in range(len(I)):
#     cur_len = len(I[i])
#     if cur_len > max_length:
#         max_length = cur_len

# masks = []

# for i in range(len(I)):
#     cur_len = len(I[i])
#     # Create an array of zeros with size n
#     mask = np.zeros(max_length, dtype=int)
#     # Set the first k elements to 1
#     mask[:cur_len] = 1
#     masks.append(mask)