import torch 
from models.gcn import GCN
from models.gat import GAT
from data.dataset import load_dataset
from training.train import  train
from evaluation.evaluate import evaluate

torch.manual_seed(123456)
dataset, data=load_dataset()
model_name=input("gcn/gat: ").lower()

if model_name=="gcn":
   model = GCN(dataset.num_features, 16, dataset.num_classes)
   lr=0.01
elif model_name=="gat":
   model = GAT(dataset.num_features, 8, dataset.num_classes, 8)
   lr=0.02
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
criterion=torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model,data,optimizer,criterion)

    print(f'Epoch: {epoch:03d}, ' f'Loss: {loss:.4f}')
val_acc = evaluate(model, data, data.val_mask)
test_acc = evaluate(model, data, data.test_mask)
print("test_acc:", test_acc)
print("val_acc:", val_acc)
