# T-SDHCAL_GNN

## Models
4 models are constructed in my_model.py

1. Transformer
2. Fully connected neural network without time
3. Fully connected neural network with time
4. Graphic neural network with time
5. Dynamic graph convolutional neural network

Transformer can be interesting, but currently it is suffering from gradient explosion and requires more debugging.
Among these models, only fully connected neural networks are working, with the error of around 0.1. However, the error does not decrease with the added time information.
GNN and DGCNN are two other potentially suitable solutions. But currently, only the input and output dimensions have been debugged to ensure they can run. The specific network structure still needs to be adjusted based on the actual physical information and the way physical information affects the results.

## To use the GNN model
### Prepare the data

1. Convert root file into npz file
    Original data is 'PionxxGeV.root' with xx: 10, 20, 30, 40, 50.
    Use the following python code to transform root file into npz file:
    ```python
    import numpy as np
    import uproot
    
    batch_size = 1000  # Define the number of samples to process in each batch (not used in this code)
    index_energy = ['10', '20', '30', '40', '50']  # Define a list of different energy values
    
    # Loop through each energy value
    for ind in index_energy:
        # Construct the file path for each ROOT file
        org_file_route = 'Your/Path/Pion' + ind + 'GeV.root'
    
        # Open the ROOT file
        file = uproot.open(org_file_route)
        
        # Get the data object named 'tree;1' from the file
        data_array = file['tree;1']
        
        # Extract the keys from the data object and convert to a NumPy array
        keys = np.array(list(data_array.keys()))
        
        # Extract the values from the data object and convert to a NumPy array
        values = np.array(list(data_array.values()))
    
        # Save the extracted keys and values as a .npz file format
        np.savez('pion' + ind + 'Gev', keys=keys, values=values)
    ```
2. Split train and test data
    Run 'data_processing_GNN.py', it will combine the data of all energy levels, shuffle them and split them into train and test data.
    4 files are generated after this step: 'data/train_indices_GNN.csv', 'data/test_indices_GNN.csv', 'data/train_data_GNN.npz', 'data/test_data_GNN.npz'
3. Generate edge index (heavy load)
    Since edge index is needed for the training of GNN, it is generated by 'dataset_preparation_GNN.py'.
    The edge index is generated by both Euclidean distance and time distance, which means only the hits that are close to each other both on position and time are connected as an edge.
    This step is very computationally intensive, but only needs to be done once. I can upload the result file to a cloud server for download if needed.
    After this step, the dataset should be 'dataset_train_directed.pt' and 'dataset_test_directed.pt', and we are ready for run.

### Train

CUDA Version: 12.7
Torch version: 2.4.0

Run 'train_GNN.py'.
