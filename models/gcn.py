import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
  def __init__(self,in_channels, hidden_channels, out_channels):
    super().__init__()
    self.conv1=GCNConv(in_channels, hidden_channels)
    self.conv2=GCNConv( hidden_channels, 10)
    self.conv3=GCNConv(10,  out_channels)
  def forward(self, x, edge_index ):
     x=self.conv1(x, edge_index)
     x=x.relu()
     # Dropout: randomly drops features during training.
     x=F.dropout(x,p=0.5, training=self.training)   # This line can be removed if we do not want to use dropout.
     x=self.conv2(x, edge_index)
     x=x.relu()
     x=F.dropout(x,p=0.5, training=self.training)
     x=self.conv3(x, edge_index)
     return x
