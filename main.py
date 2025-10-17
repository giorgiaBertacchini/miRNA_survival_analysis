from scripts.linear_simple import MiRNANet_3, MiRNANet_5, MiRNADataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


DATA_PATH = './datasets/preprocessed/final_merged_clinical_miRNA.csv'

df = pd.read_csv(DATA_PATH)

# Converti la colonna di stringhe in liste di float
def parse_array(x):
    if isinstance(x, str):
        x = x.strip("[]")
        return np.array([float(i) for i in x.split(",")])
    
    return np.array(x)

def create_age_classes(y, years_per_class=2, min_age=None):
    y = np.array(y)

    y_class = ((y - (min_age or 0)) // years_per_class).astype(int)
    return y_class

df["reads_per_million_miRNA_mapped"] = df["reads_per_million_miRNA_mapped"].apply(parse_array)
X = np.vstack(df["reads_per_million_miRNA_mapped"].values)

years_per_class=5
min_age=10
max_age=110

tot_num_classes = (max_age - min_age) // years_per_class

y = df["age_at_initial_pathologic_diagnosis"].values
y = create_age_classes(y, years_per_class=years_per_class, min_age=min_age)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = MiRNADataset(X_train, y_train)
test_ds = MiRNADataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=90)
test_loader = DataLoader(test_ds, batch_size=90)

input_dim = X_train.shape[1]
model_1 = MiRNANet_3(input_dim, tot_num_classes, start_lr=0.0001)
# model_2 = MiRNANet_5(input_dim, tot_num_classes)

model_1_res = model_1.loop(train_loader, test_loader, epochs=200)
# model_2_res = model_2.loop(epochs=150)
