import torch
import torch.nn as nn
import time

def train(device, model, train_loader, val_loader, num_epochs, l_rec=1, l_reg=1,
l_koopman=1, lr=0.001, gradclip=1, should_eval=True):
  model = model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
  reconstruction_criterion = nn.MSELoss().to(device)
  regression_criterion = nn.MSELoss().to(device)
  z_reconstruction_criterion = nn.MSELoss().to(device)
  koopman_x_reconstruction_criterion = nn.MSELoss().to(device)
  koopman_reg_crit = nn.MSELoss().to(device)

  loss_hist = {'train': {
      'rec': [],
      'reg': [],
      'koopman': [],
      'total': []
  }, 'val': {
      'rec': [],
      'reg': [],
      'koopman': [],
      'total': []
  }}

  for epoch in range(num_epochs):
    start_time = time.time()
    epoch_reconstruction_loss = 0
    epoch_regression_loss = 0
    epoch_koopman_loss = 0

    for (batch_i, batch_data) in enumerate(train_loader):
      # batch_i out of total batches
      print(f'Batch: {batch_i + 1}/{len(train_loader)}', end='\r')
      # Get data
      X, y = batch_data
      X = X.to(device)
      y = y.to(device)

      # Forward pass
      (rec, x), (z_f, z_hat_f), (x_f, x_hat_f), (regression, reg_hat) = model(X)

      reconstruction_loss = l_rec * reconstruction_criterion(X, rec)
      regression_loss = l_reg * regression_criterion(y, regression.reshape(-1))
      z_reconstruction_loss = z_reconstruction_criterion(z_f, z_hat_f)
      koopman_x_reconstruction_loss = koopman_x_reconstruction_criterion(x_f, x_hat_f)
      koopman_reg_loss = koopman_reg_crit(y, reg_hat.reshape(-1))
      total_koopman_loss = l_koopman * (z_reconstruction_loss + koopman_x_reconstruction_loss + koopman_reg_loss)

      loss = reconstruction_loss + regression_loss + total_koopman_loss

      epoch_reconstruction_loss += reconstruction_loss.item()

      epoch_regression_loss += regression_loss.item()
      epoch_koopman_loss += total_koopman_loss.item()

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), gradclip)
      optimizer.step()

    avg_rec_epoch_loss = epoch_reconstruction_loss / len(train_loader.dataset)
    avg_reg_epoch_loss = epoch_regression_loss / len(train_loader.dataset)
    avg_koopman_epoch_loss = epoch_koopman_loss / len(train_loader.dataset)

    loss_hist['train']['rec'].append(avg_rec_epoch_loss)
    loss_hist['train']['reg'].append(avg_reg_epoch_loss)
    loss_hist['train']['koopman'].append(avg_koopman_epoch_loss)
    loss_hist['train']['total'].append(avg_rec_epoch_loss + avg_reg_epoch_loss + avg_koopman_epoch_loss)

    train_end_time = time.time()
    # Validation
    if (should_eval):
      val_reconstruction_loss = 0
      val_regression_loss = 0
      val_koopman_loss = 0

      with torch.no_grad():
        for (batch_i, batch_data) in enumerate(val_loader):
          # batch_i out of total batches
          print(f'Batch: {batch_i + 1}/{len(val_loader)}', end='\r')
          # Get data
          X, y = batch_data
          X = X.to(device)
          y = y.to(device)

          # Forward pass
          (rec, x), (z_f, z_hat_f), (x_f, x_hat_f), (regression, reg_hat) = model(X)

          reconstruction_loss = l_rec * reconstruction_criterion(X, rec)
          regression_loss = l_reg * regression_criterion(y, regression.reshape(-1))
          z_reconstruction_loss = z_reconstruction_criterion(z_f, z_hat_f)
          koopman_x_reconstruction_loss = koopman_x_reconstruction_criterion(x_f, x_hat_f)
          koopman_reg_loss = koopman_reg_crit(y, reg_hat.reshape(-1))
          total_koopman_loss = l_koopman * (z_reconstruction_loss + koopman_reg_loss + koopman_x_reconstruction_loss)

          loss = reconstruction_loss + regression_loss + total_koopman_loss

          val_reconstruction_loss += reconstruction_loss.item()
          val_regression_loss += regression_loss.item()
          val_koopman_loss += total_koopman_loss.item()

        avg_val_rec_epoch_loss = val_reconstruction_loss / len(val_loader.dataset)
        avg_val_reg_epoch_loss = val_regression_loss / len(val_loader.dataset)
        avg_val_koopman_epoch_loss = val_koopman_loss / len(val_loader.dataset)

        loss_hist['val']['rec'].append(avg_val_rec_epoch_loss)
        loss_hist['val']['reg'].append(avg_val_reg_epoch_loss)
        loss_hist['val']['koopman'].append(avg_val_koopman_epoch_loss)
        loss_hist['val']['total'].append(avg_val_rec_epoch_loss + avg_val_reg_epoch_loss + avg_val_koopman_epoch_loss)

    # if (epoch == 0 or ((epoch + 1) % 5 == 0)):
    print(f'Epoch: {epoch + 1}/{num_epochs}, Time: {time.time() - start_time:.2f}s')
    print(f'\tTraining time: {train_end_time - start_time:.2f}s')
    if (should_eval):
      print(f'\tValidation time: {time.time() - train_end_time:.2f}s')
    print(f'\tReconstruction Loss: {avg_rec_epoch_loss:.4f}')
    print(f'\tRegression Loss: {avg_reg_epoch_loss:.4f}')
    print(f'\tKoopman Loss: {avg_koopman_epoch_loss:.4f}')
    print(f'\tTotal Loss: {avg_rec_epoch_loss + avg_reg_epoch_loss + avg_koopman_epoch_loss:.4f}')
    print("=====================================================")

  # Return model at no grad and eval mode
  model.eval()
  model.requires_grad_(False)

  return model, loss_hist