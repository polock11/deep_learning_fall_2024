import torch

class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, loss_func, device) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer  = optimizer
        self.loss_func = loss_func
        self.device = device
        self.model.to(device)
        
    def train_single_epoch(self):
        self.model.train()
        train_loss = 0.0
        for feature, target in self.train_loader:
            feature, target = feature.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(feature)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feature, target in self.val_loader:
                feature, target = feature.to(self.device), target.to(self.device)
                output = self.model(feature)
                loss = self.loss_func(output, target)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_single_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")