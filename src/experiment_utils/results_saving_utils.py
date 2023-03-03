import os
import json
import time
import torch

def save_grid_search_results(dataset_name, epochs, results):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    signature = f"{dataset_name}_epochs{epochs}_{timestamp}"
    all_grid_results_folder = "grid_results"
    if not os.path.exists(all_grid_results_folder):
        os.makedirs(all_grid_results_folder)
    result_folder = os.path.join(all_grid_results_folder, signature)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    # Save results as json
    json_file = os.path.join(result_folder, "results.json")
    with open(json_file, "w") as f:
        json.dump(sorted_results, f, indent=4)


def save_train_results(dataset_name, epochs, model, loss_hist, l_rec, l_reg, l_koopman, koopman_size, test_rmse):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    signature = f"{timestamp}_{dataset_name}_epochs{epochs}_lrec{l_rec}_lreg{l_reg}_lkoopman{l_koopman}_size{koopman_size}"

    all_results_folder = "experiment_results"
    if not os.path.exists(all_results_folder):
        os.makedirs(all_results_folder)
    result_folder = os.path.join(all_results_folder, signature)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save the model weights
    model_path = os.path.join(result_folder, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save training loss history
    all_loss_hist_path = os.path.join(result_folder, "all_loss_hist.json")
    with open(all_loss_hist_path, "w") as f:
        json.dump(loss_hist, f, indent=4)

    train_loss_path = os.path.join(result_folder, "train_loss.json")
    with open(train_loss_path, "w") as f:
        json.dump(loss_hist["train"], f, indent=4)

    # Save validation loss history
    val_loss_path = os.path.join(result_folder, "val_loss.json")
    with open(val_loss_path, "w") as f:
        json.dump(loss_hist["val"], f, indent=4)

    # Save test RMSE
    test_rmse_path = os.path.join(result_folder, "test_rmse.json")
    with open(test_rmse_path, "w") as f:
        json.dump(test_rmse, f, indent=4)

