import numpy as np
import torch
from sklearn.model_selection import train_test_split

def generate_data_loaders(datasetName, batch_size=32, shuffle=True):
    # Load data from .npy files
    X_train = np.load('data/npy/' + datasetName + '/X_train.npy', allow_pickle=True)
    y_train = np.load('data/npy/' + datasetName + '/y_train.npy', allow_pickle=True)
    X_test = np.load('data/npy/' + datasetName + '/X_test.npy', allow_pickle=True)
    y_test = np.load('data/npy/' + datasetName + '/y_test.npy', allow_pickle=True)

    X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_tensor, y_train_tensor, test_size=0.2)

    train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle)

    val_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle)

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle)

    return train_dataloader, val_dataloader, test_dataloader