import torch

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_func, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        for features, targets in self.train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_func(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in self.val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.loss_func(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
