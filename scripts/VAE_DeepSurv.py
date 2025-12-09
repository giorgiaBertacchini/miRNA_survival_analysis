import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ROOT = os.getcwd()
print(ROOT)
DATA_PATH = os.path.join(ROOT, 'datasets\\preprocessed')
SEED = 42
NUM_FOLDS = 10
USE_CLINICAL = True

def prepare_data(dataset, use_clinical):
    y_cols = ['Death', 'days_to_death', 'days_to_last_followup']
    if use_clinical:
        print('USING clinical data') 
        X_cols = [col for col in dataset.columns if col not in y_cols]
    else:
        print('EXCLUDING clinical data') 
        miRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'hsa' not in col]
        mRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'gene.' not in col]
        X_cols = [col for col in dataset.columns if col not in y_cols and not col in miRNA_clinical_cols and col not in mRNA_clinical_cols]

    custom_dtype = np.dtype([
        ('event', np.bool_),      # O 'bool'
        ('time', np.float64)      # O 'float'
    ])

    y = []
    for index,row in dataset[y_cols].iterrows():
        if row['Death'] == 1:
            y.append(np.array((True, row['days_to_death'].item()), dtype=custom_dtype))
        elif row['Death'] == 0:
            tuple = (False, row['days_to_last_followup'].item())
            y.append(np.array(tuple, dtype=custom_dtype)) 
    y = np.array(y)

    X = dataset[X_cols]
    return X, y    

def scale_data(X):
    scaler = StandardScaler()

    genes_cols = [col for col in X.columns if 'hsa' in col or 'gene.' in col]
    genes_cols.append('age_at_initial_pathologic_diagnosis')
    scaled_X = pd.DataFrame(scaler.fit_transform(X[genes_cols]), columns=genes_cols)

    X[genes_cols] = scaled_X
    return X

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(VAEEncoder, self).__init__()
        
        # Define the layers of the encoder
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)
        
    def reparameterize(self, mu, logvar):
        """The Reparameterization Trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from standard normal
        return mu + eps * std

    def forward(self, x):
        # Pass through dense layers
        h = self.feature_extractor(x)
        
        # Get mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize to get the latent code Z
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
class VAEDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        super(VAEDecoder, self).__init__()
        
        # Define the layers of the decoder (reverse of encoder)
        layers = []
        current_dim = latent_dim
        
        # Reverse the order of hidden_dims for the decoder
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        # Final layer to reconstruct the input dimension
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Use Tanh or Sigmoid if data is normalized, otherwise no activation (linear)
        # Assuming input gene data is normalized [0, 1] or similar
        layers.append(nn.Sigmoid()) 
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)
    
class VAECoxModel(nn.Module):
    def __init__(self, gene_dim, clinical_dim, latent_dim, vae_hidden_dims, cox_hidden_dims):
        super(VAECoxModel, self).__init__()
        
        # 1. VAE Encoder for Omics Data
        self.encoder = VAEEncoder(gene_dim, latent_dim, vae_hidden_dims)
        
        # 2. VAE Decoder (kept for combined loss calculation)
        self.decoder = VAEDecoder(gene_dim, latent_dim, vae_hidden_dims)

        # Dimension of the fused vector (Latent Z + Clinical X)
        fused_dim = latent_dim + clinical_dim 
        
        # 3. DeepSurv Head (Cox-PH predictor)
        cox_layers = []
        current_dim = fused_dim
        for h_dim in cox_hidden_dims:
            cox_layers.append(nn.Linear(current_dim, h_dim))
            cox_layers.append(nn.ReLU())
            cox_layers.append(nn.BatchNorm1d(h_dim))
            cox_layers.append(nn.Dropout(0.2)) # Use dropout for regularization
            current_dim = h_dim
            
        # Final layer outputs a single log-hazard ratio (h)
        cox_layers.append(nn.Linear(current_dim, 1))
        
        self.cox_predictor = nn.Sequential(*cox_layers)

    def forward(self, x_gene, x_clinical):
        # VAE Encoder for gene data
        z, mu, logvar = self.encoder(x_gene)
        
        # VAE Decoder for reconstruction (only needed for loss)
        x_reconstructed = self.decoder(z)
        
        # Concatenate latent Z with clinical features
        # Unsqueeze(1) is often needed for clinical data if it's 1D, 
        # but here we assume it's already batch_size x clinical_dim
        fused_input = torch.cat((z, x_clinical), dim=1)
        
        # DeepSurv prediction of log-hazard ratio (h)
        log_hazard_ratio = self.cox_predictor(fused_input)
        
        # log_hazard_ratio is batch_size x 1, so squeeze it to be 1D
        return log_hazard_ratio.squeeze(1), x_reconstructed, mu, logvar
    
class VAECoxLoss(nn.Module):
    def __init__(self, lambda_vae):
        super(VAECoxLoss, self).__init__()
        self.lambda_vae = lambda_vae # Weighting factor for VAE loss
        
    def forward(self, log_hazard_ratio, x_gene, x_reconstructed, mu, logvar, time, event):
        
        # ---------------- 1. Cox Partial Likelihood Loss ----------------
        # The Cox-PH loss is calculated using the rank ordering of event times.
        # This implementation assumes the input data is sorted by time (descending), 
        # which is standard practice in DeepSurv training loops.
        
        # The equation for the loss is: -sum_{i:event_i=1} (h_i - log(sum_{j:t_j>=t_i} exp(h_j)))
        # where h is the log-hazard ratio (log_hazard_ratio)
        
        # Calculate risk scores (exp(h))
        risk_scores = torch.exp(log_hazard_ratio)
        
        # Sum of risk scores over the risk set R_i (all patients with t_j >= t_i)
        # This is efficiently calculated using the cumulative sum on the reversed risk scores
        # because the data is assumed to be sorted by time (descending).
        
        # The cumulative sum is applied to the reversed risk scores. 
        # torch.cumsum(tensor, dim=0)
        # Note: If the tensor is already sorted by *time descending*, we can use torch.cumsum directly.
        # If the tensor is sorted by *time ascending*, we must reverse it first.
        # Assuming *time descending* (standard for DeepSurv):
        risk_set_sum = torch.cumsum(risk_scores, dim=0) 
        
        # Log of the risk set sum
        log_risk_set_sum = torch.log(risk_set_sum)
        
        # The Cox-PH loss for each patient i: h_i - log(sum_{j:t_j>=t_i} exp(h_j))
        cox_loss_numerator = log_hazard_ratio - log_risk_set_sum
        
        # Only sum the loss for patients who had an event (event=1)
        cox_loss = -torch.sum(cox_loss_numerator[event == 1])
        
        # ---------------- 2. VAE Loss (Reconstruction + KL Divergence) ----------------
        
        # A. Reconstruction Loss (e.g., Mean Squared Error for continuous gene data)
        reconstruction_loss = F.mse_loss(x_reconstructed, x_gene, reduction='sum')
        
        # B. KL Divergence Loss
        # KL[q(z|x) || p(z)] where p(z) is N(0, I) and q(z|x) is N(mu, logvar)
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - exp(log(sigma^2)))
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        
        vae_loss = reconstruction_loss + kl_divergence_loss
        
        # ---------------- 3. Total Loss ----------------
        total_loss = cox_loss + self.lambda_vae * vae_loss
        
        # We return all components for monitoring
        return total_loss, cox_loss, vae_loss
    
def main():

    datasets = [
        'miRNA\\clinical_miRNA_normalized_log.csv', 
        # 'miRNA/clinical_miRNA_normalized_quant.csv',
        # 'mRNA/clinical_mRNA_normalized_log.csv', 
        # 'mRNA/clinical_mRNA_normalized_tpm_log.csv'
    ]
    for dataset_name in datasets:
        print("Preparing data...".center(100, '-'))

        DATASET_TYPE = dataset_name.split('\\')[-1][9:-4]
        DATASET_TYPE = 'clinical_'+DATASET_TYPE if USE_CLINICAL else 'seq_only'+DATASET_TYPE
        SUBTYPE=dataset_name.split("\\")[0]

        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_name))
        X, y = prepare_data(dataset, USE_CLINICAL)
        X = scale_data(X)
        # --- 1. Define Hyperparameters ---
        gene_cols = [col for col in X.columns if 'gene.' in col or 'hsa' in col]
        clin_cols = [col for col in X.columns if col not in gene_cols]
        GENE_DIM = len(gene_cols) # Example: miRNA data (P)
        CLINICAL_DIM = len(clin_cols)
        LATENT_DIM = 50 # Compressed Z dimension (K)
        VAE_HIDDEN_DIMS = [512, 128] # Layers for the Encoder/Decoder
        COX_HIDDEN_DIMS = [64, 32] # Layers for the DeepSurv Head
        LAMBDA_VAE = 0.1 # Weighting factor for VAE loss
        LEARNING_RATE = 1e-4

        # --- 2. Instantiate Model and Loss ---
        model = VAECoxModel(GENE_DIM, CLINICAL_DIM, LATENT_DIM, VAE_HIDDEN_DIMS, COX_HIDDEN_DIMS).to(device)
        criterion = VAECoxLoss(lambda_vae=LAMBDA_VAE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # IMPORTANT: Sort the mock data by time DESCENDING for the Cox Loss calculation
        # In a real training loop, you would need to sort your batch data before each step.
        # Here we do it once for the mock data:
        sorted_indices = torch.argsort(torch.tensor(y['time'].copy()), descending=True)
        X = X.iloc[sorted_indices]
        y = y[sorted_indices]

        # X = torch.tensor(X.values)

        # --- 4. Single Training Step Example ---
        model.train()
        optimizer.zero_grad()

        # Forward Pass
        log_h, x_rec, mu, logvar = model(torch.tensor(X[gene_cols].values, dtype=torch.float32).to(device), torch.tensor(X[clin_cols].values, dtype=torch.float32).to(device))

        # Calculate Loss
        total_loss, cox_loss, vae_loss = criterion(
            log_h, torch.tensor(X[gene_cols].values, dtype=torch.float32).to(device), x_rec, mu, logvar, y['time'], y['event']
        )

        # Backward Pass and Optimization
        total_loss.backward()
        optimizer.step()

        # --- 5. Print Results ---
        print("\n--- Single Training Step Results ---")
        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"Cox-PH Loss: {cox_loss.item():.4f}")
        print(f"VAE Loss (Weighted): {(LAMBDA_VAE * vae_loss).item():.4f} (Raw: {vae_loss.item():.4f})")
        print(f"Lambda (VAE Weight): {LAMBDA_VAE}")

if __name__ == '__main__':
    main()