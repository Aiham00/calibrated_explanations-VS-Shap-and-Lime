import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from ucimlrepo import fetch_ucirepo 



# Define PyTorch Dataset class
class AagruDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Define GRU with Attention model
class GRUAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUAttention, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Attention layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1) 
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.fitted = False
        self._estimator_type = "classifier"
    def forward(self, x):
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # Weighted sum (batch, hidden_size)
        out = self.fc(context)  # Output layer
        return out
    
    def predict_proba(self, X):
        """Return class probabilities for given input"""
        self.eval()
        with torch.no_grad():
            if len(X.shape) == 2:  # If input is (batch_size, input_size), reshape it
                X = X[:, None, :]  # Add sequence length of 1 (batch_size, 1, input_size)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            probabilities = self.softmax(logits)
        return probabilities.numpy()  # Convert to NumPy for easier handling

    def predict(self, X):
        """Return the predicted class label"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)  # Get class with highest probability
    def __sklearn_is_fitted__(self):
        return self.fitted
    def get_tags(self):
        self.estimator_type = "classifier"
        return self
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, num_epochs=200, batch_size=32, learning_rate=None):
        """Train the GRU model, with optional validation"""
        
        train_dataset = AagruDataset(X_train, y_train)    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer if learning_rate is None else optim.Adam(self.parameters(), lr=learning_rate)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                

            # Optional validation
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                    val_outputs = self.forward(X_val_tensor)
                    val_preds = torch.argmax(val_outputs, axis=1).numpy()
                    val_acc = accuracy_score(y_val, val_preds)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = self.state_dict()

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Load best model only if validation was used
        if best_model_state and X_val is not None:
            self.load_state_dict(best_model_state)
            print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        self.fitted = True
        self.classes_ = np.unique(y_train)
        return self
