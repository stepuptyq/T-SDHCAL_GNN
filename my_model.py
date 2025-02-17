import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, EdgeConv, GCNConv, global_mean_pool, global_max_pool

# Hyperparameters
dim_x = 100 + 1  # Define the x dimension of input, + 1 for padding token
dim_y = 100 + 1  # Define the y dimension of input
dim_w = 48 + 1   # Define width of the channel
embedding_dim1 = 96
embedding_dim2 = 96
embedding_dim3 = 64
total_embedding_dim = embedding_dim1 + embedding_dim2 + embedding_dim3  # Total dimension after concatenation

edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

model_dim = 256 # Ensure this is suitable for the concatenated dimension if necessary
assert total_embedding_dim == model_dim
num_heads = 8
num_encoder_layers = 3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).requires_grad_(False)  # Add batch dimension

    def forward(self, x):
        return self.encoding[:, :x.size(1)] # .copy().requires_grad_(False)


# Model with multiple embeddings
class TransformerModel(nn.Module):
    def __init__(self, max_input_length):
        super(TransformerModel, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=dim_x, embedding_dim=embedding_dim1)
        self.embedding2 = nn.Embedding(num_embeddings=dim_y, embedding_dim=embedding_dim2)
        self.embedding3 = nn.Embedding(num_embeddings=dim_w, embedding_dim=embedding_dim3)
        self.pos_encoder = PositionalEncoding(d_model=model_dim, max_len=max_input_length)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.output_fc = nn.Linear(model_dim, 1)  # Energy output

    def forward(self, x1, x2, x3, mask, device):
        emb1 = self.embedding1(x1)
        emb2 = self.embedding2(x2)
        emb3 = self.embedding3(x3)
        x = torch.cat((emb1, emb2, emb3), dim=-1)  # Concatenate embeddings along the last dimension
        x = x + self.pos_encoder(x).to(device)
        # Reshape to transformer input shape
        x = x.transpose_(0,1)
        # Now shape: Seq Len x Batch Size x Input Dim
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Reshape back, now shape: Batch Size x Seq Len x Input Dim
        x = x.transpose_(0,1)
        # Use Bert-Style Pooling: only keep the first element of the seq.
        x = x[:,0,:].squeeze()
        x = self.output_fc(x)

        return x

class FullyConnectedModel(nn.Module):
    def __init__(self, max_input_length):
        super(FullyConnectedModel, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=dim_x, embedding_dim=embedding_dim1)
        self.embedding2 = nn.Embedding(num_embeddings=dim_y, embedding_dim=embedding_dim2)
        self.embedding3 = nn.Embedding(num_embeddings=dim_w, embedding_dim=embedding_dim3)

        self.encoder_layer = nn.Sequential(
            nn.Linear(total_embedding_dim * max_input_length, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True)
        )
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.output_fc = nn.Linear(model_dim, 1)  # Energy output

    def forward(self, x1, x2, x3, mask, device):
        emb1 = self.embedding1(x1)
        emb2 = self.embedding2(x2)
        emb3 = self.embedding3(x3)
        x = torch.cat((emb1, emb2, emb3), dim=-1)  # Concatenate embeddings along the last dimension
      
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_layer(x)

        x = self.output_fc(x)
        return x
    
class FullyConnectedModel_t(nn.Module):
    def __init__(self, max_input_length):
        super(FullyConnectedModel_t, self).__init__()
        # dim_t = 101
        # embedding_dim4 = 1
        self.embedding1 = nn.Embedding(num_embeddings=dim_x, embedding_dim=embedding_dim1)
        self.embedding2 = nn.Embedding(num_embeddings=dim_y, embedding_dim=embedding_dim2)
        self.embedding3 = nn.Embedding(num_embeddings=dim_w, embedding_dim=embedding_dim3)
        total_embedding_dim_t = total_embedding_dim + 1
        self.encoder_layer = nn.Sequential(
            nn.Linear(total_embedding_dim_t * max_input_length, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True)
        )
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.output_fc = nn.Linear(model_dim, 1)  # Energy output

    def forward(self, x1, x2, x3, t, mask, device):
        t_reshape = t.unsqueeze(-1)
        emb1 = self.embedding1(x1)
        emb2 = self.embedding2(x2)
        emb3 = self.embedding3(x3)
        x = torch.cat((emb1, emb2, emb3, t_reshape), dim=-1)  # Concatenate embeddings along the last dimension
      
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_layer(x)

        x = self.output_fc(x)
        return x

class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(4, 32)  # Input feature dimension is 4 (x, y, z, t), output dimension is 32
        self.conv2 = GCNConv(32, 64)  # Second graph convolution layer
        # Fully connected layers
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)  # Final output is a scalar (particle energy R)
        

    def forward(self, data):
        # Extract node features and edge index
        x = data.x           # Node feature matrix for all graphs, shape [total_num_nodes, num_node_features]
        edge_index = data.edge_index  # Edge index matrix for all graphs, shape [2, total_num_edges]
        batch = data.batch   # Index of the graph each node belongs to, shape [total_num_nodes]

        # If there are edge features, they can be extracted
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        # Graph convolution operation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Global pooling operation (feature aggregation for each graph)
        x = global_mean_pool(x, batch)  # Aggregate node features for each graph, shape [num_graphs, hidden_dim]

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output prediction
        return x

class DGCNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, k=20):
        """
        DGCNN Model
        Args:
            input_dim (int): Number of input features per node (e.g., x, y, z, t).
            output_dim (int): Output dimension (e.g., energy prediction).
            k (int): Number of nearest neighbors for graph construction.
        """
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ))
        self.conv3 = EdgeConv(nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ))

        # Fully connected layers for global feature aggregation
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, data):
        """
        Forward pass
        Args:
            data (torch_geometric.data.Data): Input data containing:
                - x: Node features [num_nodes, input_dim].
                - batch: Batch indices [num_nodes].
        Returns:
            torch.Tensor: Predicted output for each batch.
        """
        x, batch = data.x, data.batch

        # Step 1: Construct k-NN graph dynamically
        edge_index = knn_graph(x, k=self.k, batch=batch)

        # Step 2: Apply EdgeConv layers
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Step 3: Global pooling (max pooling over nodes in each batch)
        x_global = global_max_pool(x3, batch)

        # Step 4: Fully connected layers for prediction
        x = F.relu(self.fc1(x_global))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x