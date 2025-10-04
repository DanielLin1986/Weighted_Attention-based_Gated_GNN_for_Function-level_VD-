import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch_geometric.transforms as geom_transforms

from typing import Union

class GGNNFlatSum(nn.Module):
    """ Graph level prediction with gated graph recurrent networks
    """

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 num_layers:int,
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs) -> None:
        """
        Args:
            input_channels - Dimension of the input features
            hidden_channels - Dimension of hidden features
            num_layers - Number of layers for the Recurrent Graph Network
            aggr - Aggregation method used for the recurrent graph network
            bias - If set to false, the recurrent network will not learn an additive bias
            kwargs - Additional arguments for the GatedGraphConv model
        """
        super().__init__()
        self.ggnn = geom_nn.GatedGraphConv(out_channels=hidden_channels,
                                           num_layers=num_layers, aggr=aggr,
                                           bias=bias, **kwargs)
        # MLP
        self.head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(input_channels + hidden_channels, 1),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges
                in the graph (Pytorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        out = self.ggnn(x=x, edge_index=edge_index)
        x = torch.cat((x, out), axis=1)
        x = self.head(x)
        x = geom_nn.global_add_pool(x, batch_index)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x

class WARVD(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, edge_attr_dim: int, aggr='mean', **kwargs):
        super().__init__()
        # 1. Edge encoding
        self.edge_encoder = nn.Sequential(
            nn.Embedding(edge_attr_dim, hidden_channels // 4),
            nn.Linear(hidden_channels // 4, hidden_channels),
            nn.GELU()
        )

        # 2. Projection of the input
        self.input_proj = nn.Linear(input_channels, hidden_channels)

        # 3. Gated and multi-layer convolution
        self.convs = nn.ModuleList([
            geom_nn.GatedGraphConv(hidden_channels, num_layers=num_layers, aggr=aggr)
            for _ in range(num_layers)
        ])

        # 4. Attention mechanism
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels), # * 3
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # 4. Linear layers
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.batchNor = nn.BatchNorm1d(hidden_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 5. Learnable weight parameters
        self.loss_weight = nn.Parameter(torch.tensor([1.0, 5.0]), requires_grad=True)  # Initialized weights tensor 5.0x more on vulnerable class  (Enabled in real-world dataset)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # === Feature encoding ===
        #  [E, hidden]
        x = self.input_proj(x)
        edge_emb = self.edge_encoder(edge_attr.squeeze())

        # === Layered message passing ===
        h = x
        for conv in self.convs:
            # Edge messages
            h_new = conv(h, edge_index)

            # Edge attention [E,1]
            src, dst = edge_index
            attn_input = torch.cat([h[src], h[dst], edge_emb], dim=1)
            edge_attn = self.edge_attention(attn_input)

            # Meaning pooling [N, hidden]
            weighted_msg = geom_nn.global_mean_pool(edge_attn * edge_emb, dst)
            h = h_new + weighted_msg  # residual connection

        # === Classification ===
        # Concatenate the processed feature h with original features  [N, input+hidden]
        combined = torch.cat([x, h], dim=1) #

        # Meaning pooling [B, input+hidden]
        graph_feat = geom_nn.global_mean_pool(combined, batch)

        # First Linear
        fc1_output = self.fc1(graph_feat)
        batch_Nor = self.batchNor(fc1_output)

        mlp_feature = self.gelu(batch_Nor) # Use the second last hidden layer's output as the high-level representations.
        fc2_output = self.fc2(mlp_feature)
        out_feature = self.sigmoid(fc2_output).squeeze(1)

        return out_feature, mlp_feature
