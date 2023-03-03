import torch
import numpy as np

def eval(device, model, test_loader):
  model = model.to(device)
  model.eval()
  total_loss = 0
  count = 0

  total_loss2 = 0
  with torch.no_grad():
    for (batch_i, batch_data) in enumerate(test_loader):
      print(f'Eval Batch: {batch_i + 1}/{len(test_loader)}', end='\r')
      # Get data
      X, y = batch_data
      X = X.to(device)
      y = y.to(device)

      # Forward pass
      (rec, x), (z_f, z_hat_f), (x_f, x_hat_f), (regression, reg_hat) = model(X)

      loss = torch.nn.functional.mse_loss(regression.reshape(-1), y, reduction='sum')

      total_loss += loss.item()
      count += X.size(0)

    rmse = np.sqrt(total_loss / count)
    print("RMSE: ", rmse)

    return rmse
