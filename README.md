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
Original data is 'PionxxGeV.root' with xx: 10, 20, 30, 40, 50.
Use the following python code to transform root file into npz file:
'''python
import numpy as np  # Import NumPy library for numerical computations and array operations
import uproot  # Import uproot library for reading ROOT file format data

batch_size = 1000  # Define the number of samples to process in each batch (not used in this code)
index_energy = ['10', '20', '30', '40', '50']  # Define a list of different energy values

# Loop through each energy value
for ind in index_energy:
    # Construct the file path for each ROOT file
    org_file_route = 'E:/SJTU/phd/GNN/Pion' + ind + 'GeV.root'

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

ds
