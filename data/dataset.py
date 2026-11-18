from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
def load_dataset():
  project_root=Path(__file__).resolve().parents[1]
  data_path=project_root/"data"/"Cora"
  dataset=Planetoid(root=str(data_path), name="Cora", transform=NormalizeFeatures())
  data=dataset[0]
  return dataset, data
