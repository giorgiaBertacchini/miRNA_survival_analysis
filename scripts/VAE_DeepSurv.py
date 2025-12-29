import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
import itertools
import json
import os
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

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
        # layers.append(nn.Tanh()) 
        
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
        
    def forward(self, log_hazard_ratio, x_gene, x_reconstructed, mu, logvar, time, event, beta_kl=1.0):
        
        # ---------------- 1. Cox Partial Likelihood Loss ----------------
        # The Cox-PH loss is calculated using the rank ordering of event times.
        # This implementation assumes the input data is sorted by time (descending), 
        # which is standard practice in DeepSurv training loops.
        
        risk_scores = torch.exp(log_hazard_ratio)
        risk_set_sum = torch.cumsum(risk_scores, dim=0) 

        # Log of the risk set sum
        log_risk_set_sum = torch.log(risk_set_sum)
        
        # The Cox-PH loss for each patient i: h_i - log(sum_{j:t_j>=t_i} exp(h_j))
        cox_loss_numerator = log_hazard_ratio - log_risk_set_sum
        events = (event == 1)
        if events.any():
            # Mean loss per event for scale stability
            cox_loss = -torch.mean(cox_loss_numerator[events])
        else:
            cox_loss = torch.tensor(0.0, device=log_hazard_ratio.device)
        
        # ---------------- 2. VAE Loss (Reconstruction + KL Divergence) ----------------
        
        # A. Reconstruction Loss (e.g., Mean Squared Error for continuous gene data)
        reconstruction_loss = F.mse_loss(x_reconstructed, x_gene, reduction='mean')
        
        # B. KL Divergence Loss
        kl_divergence_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
        
        vae_loss = reconstruction_loss + (beta_kl * kl_divergence_loss)
        
        # ---------------- 3. Total Loss ----------------
        total_loss = cox_loss + (self.lambda_vae * vae_loss)
        
        # We return all components for monitoring
        return total_loss, cox_loss, vae_loss
    
def main():
    datasets = [
        'miRNA\\clinical_miRNA_normalized_log.csv', 
        'miRNA\\clinical_miRNA_normalized_quant.csv',
        'mRNA\\clinical_mRNA_normalized_log.csv', 
        'mRNA\\clinical_mRNA_normalized_tpm_log.csv'
    ]

    latent_dims = [50]
    vae_hidden_dims_list = [[256, 64], [512, 128]]
    cox_hidden_dims_list = [[32, 16], [64, 32]]
    lambda_vaes = [0.7] #, 0.8, 0.5] # [0.001, 0.005, 0.01]
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [32, 64, 128]

    grid = list(itertools.product(latent_dims, vae_hidden_dims_list, cox_hidden_dims_list, lambda_vaes, learning_rates, batch_sizes))

    for dataset_name in datasets:
        print("Preparing data...".center(100, '-'))

        DATASET_TYPE = dataset_name.split('\\')[-1][9:-4]
        DATASET_TYPE = 'clinical_'+DATASET_TYPE if USE_CLINICAL else 'seq_only'+DATASET_TYPE
        SUBTYPE = dataset_name.split("\\")[0]

        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_name))
        X, y = prepare_data(dataset, USE_CLINICAL)
        X = scale_data(X)

        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)

        gene_cols = [col for col in X.columns if 'gene.' in col or 'hsa' in col]
        clin_cols = [col for col in X.columns if col not in gene_cols]
        GENE_DIM = len(gene_cols)
        CLINICAL_DIM = len(clin_cols)

        sorted_indices = torch.argsort(torch.tensor(y['time'].copy()), descending=True)
        X = X.iloc[sorted_indices].reset_index(drop=True)
        y = y[sorted_indices]

        X_gene_tensor = torch.tensor(X[gene_cols].values, dtype=torch.float32)
        X_clin_tensor = torch.tensor(X[clin_cols].values, dtype=torch.float32)
        time_tensor = torch.tensor(y['time'].copy(), dtype=torch.float32)
        event_tensor = torch.tensor(y['event'].copy(), dtype=torch.bool)

        X_gene_train, X_clin_train = X_gene_tensor[train_idx], X_clin_tensor[train_idx]
        time_train, event_train = time_tensor[train_idx], event_tensor[train_idx]
        X_gene_val, X_clin_val = X_gene_tensor[val_idx], X_clin_tensor[val_idx]
        time_val, event_val = time_tensor[val_idx], event_tensor[val_idx]

        params_dir = f"grid_searches/vae/{SUBTYPE}/{DATASET_TYPE}"
        os.makedirs(params_dir, exist_ok=True)
        params_path = f"{params_dir}/best_params.json"

        best_c_index = -1.0
        best_params = None
        best_model_state = None

        # Check if best_params.json exists
        if os.path.exists(params_path):
            print(f"Loading best parameters from {params_path}")
            with open(params_path, "r") as f:
                best_params = json.load(f)
            LATENT_DIM = best_params["latent_dim"]
            VAE_HIDDEN_DIMS = best_params["vae_hidden_dims"]
            COX_HIDDEN_DIMS = best_params["cox_hidden_dims"]
            LAMBDA_VAE = best_params["lambda_vae"]
            LEARNING_RATE = best_params["learning_rate"]
            BATCH_SIZE = best_params["batch_size"]
        else:
            best_val_loss = float('inf')

            for LATENT_DIM, VAE_HIDDEN_DIMS, COX_HIDDEN_DIMS, LAMBDA_VAE, LEARNING_RATE, BATCH_SIZE in grid:
                print(f"\nGrid Search: latent_dim={LATENT_DIM}, vae_hidden={VAE_HIDDEN_DIMS}, cox_hidden={COX_HIDDEN_DIMS}, "
                      f"lambda_vae={LAMBDA_VAE}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")

                model = VAECoxModel(GENE_DIM, CLINICAL_DIM, LATENT_DIM, VAE_HIDDEN_DIMS, COX_HIDDEN_DIMS).to(device)
                criterion = VAECoxLoss(lambda_vae=LAMBDA_VAE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                train_data = TensorDataset(X_gene_train, X_clin_train, time_train, event_train)
                dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
                # dataset_torch = TensorDataset(X_gene_tensor, X_clin_tensor, time_tensor, event_tensor)
                # dataloader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)

                EPOCHS = 30  # Reduced for grid search speed

                cox_losses = []
                vae_losses = []
                total_losses = []

                for epoch in range(EPOCHS):
                    model.train()
                    epoch_loss = 0
                    epoch_cox_loss = 0
                    epoch_vae_loss = 0
                    num_batches = 0
                    beta = min(1.0, epoch / (EPOCHS//2))
                    for X_gene_batch, X_clin_batch, time_batch, event_batch in dataloader:
                        sorted_idx = torch.argsort(time_batch, descending=True)
                        X_gene_batch = X_gene_batch[sorted_idx].to(device)
                        X_clin_batch = X_clin_batch[sorted_idx].to(device)
                        time_batch = time_batch[sorted_idx].to(device)
                        event_batch = event_batch[sorted_idx].to(device)

                        optimizer.zero_grad()
                        log_h, x_rec, mu, logvar = model(X_gene_batch, X_clin_batch)
                        total_loss, cox_loss, vae_loss = criterion(
                            log_h, X_gene_batch, x_rec, mu, logvar, time_batch, event_batch, beta_kl=beta
                        )
                        total_loss.backward()
                        # Gradient clipping prevents the Cox loss from exploding the weights
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        epoch_loss += total_loss.item()
                        epoch_cox_loss += cox_loss.item()
                        epoch_vae_loss += vae_loss.item()
                        num_batches += 1
                    cox_losses.append(epoch_cox_loss / num_batches)
                    vae_losses.append(epoch_vae_loss / num_batches)
                    total_losses.append(epoch_loss / num_batches)

                # Compute the mean log10 scale difference between cox and vae losses over all epochs
                # (lower is better, 0 means same order of magnitude)
                model.eval()
                with torch.no_grad():
                    # Get predictions on validation set
                    # Ensure validation data is on the correct device
                    val_x_g = X_gene_val.to(device)
                    val_x_c = X_clin_val.to(device)
                    val_t = time_val.to(device)
                    val_e = event_val.to(device)

                    val_log_h, val_rec, val_mu, val_logvar = model(val_x_g, val_x_c)
                    
                    # 1. Calculate Validation Losses using the VAECoxLoss criterion
                    # Use beta_kl=1.0 for evaluation to get the full ELBO/Total Loss
                    val_total_loss, val_cox, val_vae = criterion(
                        val_log_h, val_x_g, val_rec, val_mu, val_logvar, val_t, val_e, beta_kl=beta
                    )
                    
                    val_total_loss = val_total_loss.item()
                    val_cox = val_cox.item()
                    val_vae = val_vae.item()

                    # 2. (Optional) Still calculate C-Index and MSE for monitoring
                    val_c_index = concordance_index_censored(event_val.numpy(), time_val.numpy(), -val_log_h.cpu().numpy())[0]
                    val_mse = F.mse_loss(val_rec, val_x_g).item()

                print(f"Val Total Loss: {val_total_loss:.4f} (Cox: {val_cox:.4f}, VAE: {val_vae:.4f}) | C-Index: {val_c_index:.4f} | MSE: {val_mse:.4f}")

                # --- SELECTION CRITERION: MINIMIZE TOTAL VALIDATION LOSS ---
                # This selects the model that best balances reconstruction and survival prediction
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    best_params = {
                        "latent_dim": LATENT_DIM,
                        "vae_hidden_dims": VAE_HIDDEN_DIMS,
                        "cox_hidden_dims": COX_HIDDEN_DIMS,
                        "lambda_vae": LAMBDA_VAE,
                        "learning_rate": LEARNING_RATE,
                        "batch_size": BATCH_SIZE
                    }
                            
                # WORKING: Select best params based on loss scale difference and value
                # scale_diffs = [abs(np.log10(abs(c)+1e-8) - np.log10(abs(v)+1e-8)) for c, v in zip(cox_losses, vae_losses)]
                # mean_scale_diff = np.mean(scale_diffs)
                # final_total_loss = total_losses[-1]

                # print(f"Mean log10 scale diff (Cox vs VAE): {mean_scale_diff:.3f}, Final total loss: {final_total_loss:.3f}")

                # # Select best: prioritize lowest scale diff, then lowest total loss
                # if (mean_scale_diff < 0.5 and final_total_loss < best_val_loss) or \
                #    (mean_scale_diff < best_scale_diff) or \
                #    (mean_scale_diff == best_scale_diff and final_total_loss < best_val_loss):
                #     best_val_loss = final_total_loss
                #     best_scale_diff = mean_scale_diff
                #     best_params = {
                #         "latent_dim": LATENT_DIM,
                #         "vae_hidden_dims": VAE_HIDDEN_DIMS,
                #         "cox_hidden_dims": COX_HIDDEN_DIMS,
                #         "lambda_vae": LAMBDA_VAE,
                #         "learning_rate": LEARNING_RATE,
                #         "batch_size": BATCH_SIZE
                #     }
                #     best_model_state = model.state_dict()

            # Save best parameters
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"Best parameters saved to {params_path}")

        # --- Retrain and Save VAE Embeddings with Best Parameters ---
        print(f"\nBest parameters: latent_dim={best_params['latent_dim']}, vae_hidden={best_params['vae_hidden_dims']}, "
              f"cox_hidden={best_params['cox_hidden_dims']}, lambda_vae={best_params['lambda_vae']}, "
              f"lr={best_params['learning_rate']}, batch_size={best_params['batch_size']}, C-Index={best_c_index:.4f}")
        print(f"Retraining VAE with best parameters to save final dataset...")

        LATENT_DIM = best_params["latent_dim"]
        VAE_HIDDEN_DIMS = best_params["vae_hidden_dims"]
        COX_HIDDEN_DIMS = best_params["cox_hidden_dims"]
        LAMBDA_VAE = best_params["lambda_vae"]
        LEARNING_RATE = best_params["learning_rate"]
        BATCH_SIZE = best_params["batch_size"]

        model = VAECoxModel(GENE_DIM, CLINICAL_DIM, LATENT_DIM, VAE_HIDDEN_DIMS, COX_HIDDEN_DIMS).to(device)
        criterion = VAECoxLoss(lambda_vae=LAMBDA_VAE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        dataset_torch = TensorDataset(X_gene_tensor, X_clin_tensor, time_tensor, event_tensor)
        dataloader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)

        EPOCHS = 100 # big for final training
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            epoch_cox_loss = 0
            epoch_vae_loss = 0
            num_batches = 0
            beta = min(1.0, epoch / (EPOCHS // 2))
            for X_gene_batch, X_clin_batch, time_batch, event_batch in dataloader:
                sorted_idx = torch.argsort(time_batch, descending=True)
                X_gene_batch = X_gene_batch[sorted_idx].to(device)
                X_clin_batch = X_clin_batch[sorted_idx].to(device)
                time_batch = time_batch[sorted_idx].to(device)
                event_batch = event_batch[sorted_idx].to(device)

                optimizer.zero_grad()
                log_h, x_rec, mu, logvar = model(X_gene_batch, X_clin_batch)
                total_loss, cox_loss, vae_loss = criterion(
                    log_h, X_gene_batch, x_rec, mu, logvar, time_batch, event_batch, beta_kl=beta
                )
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                epoch_cox_loss += cox_loss.item()
                epoch_vae_loss += vae_loss.item()
                num_batches += 1
            print(
                f"Epoch {epoch+1}/{EPOCHS} - "
                f"Total: {epoch_loss/num_batches:.4f} | "
                f"Cox: {epoch_cox_loss/num_batches:.4f} | "
                f"VAE: {epoch_vae_loss/num_batches:.4f} | "
                f"lambda_VAE: {LAMBDA_VAE}"
            )

        model.eval()
        with torch.no_grad():
            X_gene_tensor_full = torch.tensor(X[gene_cols].values, dtype=torch.float32).to(device)
            z, _, _ = model.encoder(X_gene_tensor_full)
            z = z.cpu().numpy()
            clinical_data = X[clin_cols].reset_index(drop=True)
            embeddings_df = pd.DataFrame(z, columns=[f"VAE_{i}" for i in range(z.shape[1])])
            final_df = pd.concat([embeddings_df, clinical_data, dataset[['Death', 'days_to_death', 'days_to_last_followup']]], axis=1)
            fname = os.path.join(ROOT, f"datasets/preprocessed/{SUBTYPE}/VAE_{DATASET_TYPE}.csv")
            final_df.to_csv(fname, index=False)
            print(f"VAE embeddings with clinical data saved to {fname}")

if __name__ == '__main__':
    main()