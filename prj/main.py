import torch
from torch.utils.data import DataLoader
from data_preprocessor import DataPreprocessor
from dataset import HousingDataset
from model import HousingModel
from trainer import Trainer
import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import fetch_california_housing


if __name__ == "__main__":
    # Configuration
    train_size = 0.7
    test_size = 0.15
    val_size = 0.15
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

    # Data preparation
    data = fetch_california_housing()
    X, y = data.data, data.target
    preprocessor = DataPreprocessor(X, y, train_size, test_size, val_size)
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.preprocess()

    train_dataset = HousingDataset(X_train, y_train)
    test_dataset = HousingDataset(X_test, y_test)
    val_dataset = HousingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    input_dim = X_train.shape[1]
    model = HousingModel(input_dim)
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training
    trainer = Trainer(model, train_loader, val_loader, loss_func, optimizer, device)
    trainer.fit(epochs)
