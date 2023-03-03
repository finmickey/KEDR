import torch

def normalize(train_dataloader, val_dataloader, test_dataloader):
    trainX = train_dataloader.dataset.tensors[0]
    mean = torch.mean(torch.mean(trainX, 2, True), 0, True)
    std = torch.mean(torch.std(trainX, 2, keepdim=True), 0, True)
    trainX = (trainX - mean) / std
    train_dataloader.dataset.tensors = (trainX, train_dataloader.dataset.tensors[1])

    valX = val_dataloader.dataset.tensors[0]
    valX = (valX - mean) / std
    val_dataloader.dataset.tensors = (valX, val_dataloader.dataset.tensors[1])

    testX = test_dataloader.dataset.tensors[0]
    testX = (testX - mean) / std
    test_dataloader.dataset.tensors = (testX, test_dataloader.dataset.tensors[1])

    return train_dataloader, val_dataloader, test_dataloader, mean, std