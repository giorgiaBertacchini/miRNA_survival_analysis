import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

DATA_PATH = 'D:/Universita/2 anno magistrale/Progetto BioInf/miRNA_to_age/final_merged_clinical_miRNA.csv'

df = pd.read_csv(DATA_PATH)

# print(df.shape)
# print(df.head())
# print(df.iloc[0]['reads_per_million_miRNA_mapped'])

# Converti la colonna di stringhe in liste di float
def parse_array(x):
    if isinstance(x, str):
        x = x.strip("[]")
        return np.array([float(i) for i in x.split(",")])
    return np.array(x)

df["reads_per_million_miRNA_mapped"] = df["reads_per_million_miRNA_mapped"].apply(parse_array)
X = np.vstack(df["reads_per_million_miRNA_mapped"].values)
y = df["age_at_initial_pathologic_diagnosis"].values

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# print(X_scaled[0])
# print(X_scaled.std(axis=0))
# print(X_scaled.mean(axis=0))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MiRNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Porta a vettore colonna
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MiRNADataset(X_train, y_train)
test_ds = MiRNADataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32)

class MiRNANet(nn.Module):
    def __init__(self, input_dim):
        super(MiRNANet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # <--- ultimo layer lineare
        )
    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = MiRNANet(input_dim)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(150):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        # print(outputs)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Valutazione
model.eval()
with torch.no_grad():
    preds = []
    trues = []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds.append(outputs.numpy())
        trues.append(y_batch.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)

mae = np.mean(np.abs(preds - trues))
print(f"Mean Absolute Error: {mae:.2f}")