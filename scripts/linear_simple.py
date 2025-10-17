import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


class MiRNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Porta a vettore colonna
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MiRNANet_3(nn.Module):
    def __init__(self, input_dim, output_dim, start_lr=0.001):
        super(MiRNANet_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)  # <--- ultimo layer lineare
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',        # riduce LR quando la metrica (loss) smette di diminuire
            factor=0.6,        # dimezza il learning rate
            patience=15        # aspetta 10 epoche senza miglioramenti
        )
    def forward(self, x):
        return self.model(x)   
    def loop(self, train_loader, test_loader, epochs=100):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                # print(outputs)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.scheduler.step(avg_train_loss)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                preds = []
                trues = []
                for X_batch, y_batch in test_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    preds.append(outputs.numpy())
                    trues.append(y_batch.numpy())
                preds = np.vstack(preds)
                trues = np.vstack(trues)
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(test_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            mae = mean_absolute_error(trues, np.argmax(preds, axis=1))
            print(f"Test MAE: {mae:.2f}")

        model_res = {
            'params':self.model.parameters(),
            'train_losses':train_losses,
            'eval_losses': val_losses
        }

        return model_res
    
class MiRNANet_5(nn.Module):
    def __init__(self, input_dim, output_dim, start_lr = 0.001):
        super(MiRNANet_5, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)  # <--- ultimo layer lineare
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',        # riduce LR quando la metrica (loss) smette di diminuire
            factor=0.6,        # dimezza il learning rate
            patience=15        # aspetta 10 epoche senza miglioramenti
        )
    def forward(self, x):
        return self.model(x)
    
    def loop(self, train_loader, test_loader, epochs=100):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                # print(outputs)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.scheduler.step(avg_train_loss)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                preds = []
                trues = []
                for X_batch, y_batch in test_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    preds.append(outputs.numpy())
                    trues.append(y_batch.numpy())
                preds = np.vstack(preds)
                trues = np.vstack(trues)
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(test_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        model_res = {
            'params':self.model.parameters(),
            'train_losses':train_losses,
            'eval_losses': val_losses
        }

        return model_res