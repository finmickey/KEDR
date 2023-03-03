from data_utils.data_loaders import generate_data_loaders
from experiment_utils.results_saving_utils import save_grid_search_results
import torch
from eval import eval
from model import EncoderDecoder
from train import train
import time

def grid_search(device, dataset_name, batch_size, epochs=7):
  koopman_sizes = [40, 120, 200]
  ls_rec = [1, 5]
  ls_reg = [1, 5]
  ls_koopman = [0, 1, 5]

  train_dataloader, val_dataloader, test_dataloader = generate_data_loaders(dataset_name, batch_size)

  t = train_dataloader.dataset.tensors[0].float()
  mean = torch.mean(torch.mean(t, 2, True), 0, True)
  std = torch.mean(torch.std(t, 2, keepdim=True), 0, True)
  t = (t - mean) / std
  train_dataloader.dataset.tensors = (t, train_dataloader.dataset.tensors[1].float())


  t2 = val_dataloader.dataset.tensors[0].float()
  t2 = (t2 - mean) / std
  val_dataloader.dataset.tensors = (t2, val_dataloader.dataset.tensors[1].float())

  t3 = test_dataloader.dataset.tensors[0].float()
  t3 = (t3 - mean) / std
  test_dataloader.dataset.tensors = (t3, test_dataloader.dataset.tensors[1].float())

  # Create dictionary to save results
  results = {}
  results_t = {}

  # Iterate over all possibilities of parameters
  for koopman_size in koopman_sizes:
    for l_rec in ls_rec:
      for l_reg in ls_reg:
        for l_koopman in ls_koopman:
            start_time = time.time()
            print("=========================================")
            print(f"Training model with koopman_size={koopman_size}, l_rec={l_rec}, l_reg={l_reg}, l_koopman={l_koopman}")
            print("=========================================")
            model = EncoderDecoder(t.shape[1], koopman_size, t.shape[2])
            # Train model
            model, _ = train(device, model, train_dataloader, val_dataloader, epochs, l_rec, l_reg, l_koopman, should_eval=False)
            # Evaluate model
            rmse = eval(device, model, val_dataloader)
            # Save results
            results[(koopman_size, l_rec, l_reg, l_koopman)] = rmse

            rmse_t = eval(device, model, test_dataloader)
            results_t[(koopman_size, l_rec, l_reg, l_koopman)] = rmse_t

            print("=========================================")
            print(f"Finished training model with koopman_size={koopman_size}, l_rec={l_rec}, l_reg={l_reg}, l_koopman={l_koopman} in {time.time() - start_time:.2f} seconds")
            print("=========================================")

  # Format results and print
  save_grid_search_results(dataset_name, epochs, results)
  save_grid_search_results(f"{dataset_name}_test", epochs, results_t)
  return results
