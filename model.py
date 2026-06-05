import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINMultiTask(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_dim=128,
        num_layers=3,
        num_tasks=37,
        dropout=0.1
    ):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(nn1)

        self.convs = nn.ModuleList()

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        self.bn = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = dropout

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_tasks)
        ])

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn[0](x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.bn[i + 1](x)

        batch = data.batch

        g = global_add_pool(x, batch)

        g_mean = g.mean(dim=1, keepdim=True)
        g_std = g.std(dim=1, keepdim=True) + 1e-6

        g = (g - g_mean) / g_std

        g = F.dropout(
            g,
            p=self.dropout,
            training=self.training
        )

        outs = [head(g).view(-1) for head in self.heads]

        return torch.stack(outs, dim=1)