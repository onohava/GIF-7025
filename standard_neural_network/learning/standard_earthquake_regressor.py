import torch
import torch.nn as nn
import torch.optim as optim
import math 

class StandardEarthquakeRegressor(nn.Module):
    def __init__(self, input_dimensions, learning_rate):
        super().__init__()

        # Multi-layer Perceptron architecture
        self.model = nn.Sequential(
            nn.Linear(input_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.training_losses = []

    def forward(self, x):
        return self.model(x)
    
    def train_model(self, X_train, y_train, epochs=50):
        X_train = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()

            predictions = self(X_train)
            loss = self.criterion(predictions, y_train)

            loss.backward()
            self.optimizer.step()

            self.training_losses.append(loss.item())
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def evaluate(self, X_test, y_test):
        X_test = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)
        
        self.eval()
        with torch.no_grad():
            predictions = self(X_test)
            mse_loss = self.criterion(predictions, y_test)

        average_loss = math.sqrt(mse_loss.item())
        print(f"Average loss: {average_loss:.4f}")
        return mse_loss.item()
    
    def predict(self, X):
        X = torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            return self(X).numpy()
