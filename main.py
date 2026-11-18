import torch 
from models.gcn import GCN
from models.gcn import GAT
from data.dataset import load_dataset
from training.train import  train
from evaluation.evaluate import evaluate

torch.manual_seed(123456)
