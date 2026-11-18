import torch
@torch.no_grad()
def evaluate(model, data, mask):
  model.eval()
  out=model(data.x, data.edge_index)
  pred=out.argmax(dim=1)
  correct=(pred[mask]==data.y[mask])
  acc=(int(correct.sum())/int(mask.sum()))
  return acc
