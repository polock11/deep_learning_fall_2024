from sklearn.datasets import fetch_california_housing
from data_preparation import DataPreparation
from dataset import MakeDataset
from model import Model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import torch

if __name__ == "__main__":

    train_size = .7
    test_size = .15
    val_size = .15
    learning_rate = 0.001
    eopchs = 50
    batch_size = 64
    device = (
    "cuda"
    if torch.cuda.is_available()
    #else "mps"
    #if torch.backends.mps.is_available()
    else "cpu"
    )

    data = fetch_california_housing()
    X, y = data.data, data.target
    
    data_prapare = DataPreparation(X, y, train_size, test_size, val_size)
    X_train, X_test, X_val, y_train, y_test, y_val = data_prapare.prepare()

    train_tensors = MakeDataset(X_train, y_train)
    test_tensors = MakeDataset(X_test, y_test)
    val_tensors = MakeDataset(X_val, y_val)


    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensors, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = Model(input_dim=input_dim)

    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device
    )
    trainer.fit(epochs=eopchs)
