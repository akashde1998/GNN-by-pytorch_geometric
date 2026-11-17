import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):

  def __init__(self,in_channels,hidden_channels,out_channels,heads):
   super().__init__()
   self.conv1=GATConv(in_channels, hidden_channels, heads=heads,dropout=0.6, concat=True)
   self.conv2=GATConv(hidden_channels*heads, out_channels, heads=1,dropout=0.6, concat=False)

  def forward(self, x, edge_index):
    x = F.dropout(x, p=0.6, training=self.training)
    x=self.conv1(x,edge_index)
    x=F.elu(x)
    x=F.dropout(x,p=0.6, training=self.training)
    x=self.conv2(x,edge_index)
    return x
