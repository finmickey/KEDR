from data_utils.data_loaders import generate_data_loaders
from experiment_utils.results_saving_utils import save_train_results
from data_utils.normalize import normalize
from model import EncoderDecoder
from train import train
from eval import eval


def run_experiment(dataset_name, batch_size, epochs, device, l_rec = 1, l_reg = 1, l_koopman = 1,
 koopman_size = 80, should_normalize = True):
    # Load data
    train_dataloader, val_dataloader, test_dataloader = generate_data_loaders(dataset_name, batch_size)
    if (should_normalize):
        print("Normalizing data")
        train_dataloader, val_dataloader, test_dataloader, mean, std = normalize(train_dataloader, val_dataloader, test_dataloader)

    model = EncoderDecoder(test_dataloader.dataset.tensors[0].shape[1], koopman_size, test_dataloader.dataset.tensors[0].shape[2])

    model, loss_hist = train(device, model, train_dataloader, val_dataloader, epochs, l_rec, l_reg, l_koopman)

    test_rmse = eval(device, model, test_dataloader)

    save_train_results(dataset_name, epochs, model, loss_hist, l_rec, l_reg, l_koopman, koopman_size, test_rmse)