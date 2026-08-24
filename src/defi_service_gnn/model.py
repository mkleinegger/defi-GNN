from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import BatchNorm, SAGEConv


class GraphSAGE(nn.Module):
    """The two-hop GraphSAGE classifier used by the original training script."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels, aggr="mean")
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, output_channels, aggr="mean")
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)
