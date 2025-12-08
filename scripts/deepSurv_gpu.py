import random
import os
from functools import reduce
from scipy.interpolate import interp1d

import numpy as np
import operator
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import matplotlib.pyplot as plt

from networks import Net_3layers, Net_5layers

# Fix for scipy version compatibility
import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
    
# ---------------------- CONFIG ----------------------
NUM_FOLDS = 5  # TODO 10
SEED = 42
USE_CLINICAL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_COMPONENTS = 50  # TODO Aumentare per mRNA

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # False
torch.use_deterministic_algorithms(True)  # False

# Paths
base = os.path.basename(os.getcwd())
list_path = os.getcwd().split(os.sep)
list_path.pop()
# list.pop(list.index(base))
ROOT = '\\'.join(list_path)
DATA_PATH = os.path.join(ROOT, 'datasets/preprocessed')


# ---------------------- DATA -----------------------
def prepare_data(dataset, use_clinical=USE_CLINICAL):
    y_cols = ['Death', 'days_to_last_followup', 'days_to_death']
    y = dataset[['Death', 'days_to_last_followup']].copy()
    y = y.rename(columns={'Death': 'event', 'days_to_last_followup': 'duration'})

    if use_clinical:
        print('INCLUDING clinical data')
        X_cols = [col for col in dataset.columns if col not in y_cols]
    else:
        print('EXCLUDING clinical data')
        miRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'hsa' not in col]
        mRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'gene.' not in col]
        X_cols = [col for col in dataset.columns if
                  col not in y_cols and not col in miRNA_clinical_cols and col not in mRNA_clinical_cols]

    X = dataset[X_cols].copy()
    return X, y


def scale_data(X):
    scaler = StandardScaler()
    gene_cols = [col for col in X.columns if 'hsa' in col or 'gene.' in col]
    gene_cols.append('age_at_initial_pathologic_diagnosis')
    X[gene_cols] = scaler.fit_transform(X[gene_cols])
    return X


def data_to_gpu(X_train, y_train, X_test, y_test):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32, device=DEVICE)
    y_train_t = (torch.tensor(y_train['duration'].to_numpy(dtype='float32'), device=DEVICE),
                 torch.tensor(y_train['event'].to_numpy(dtype='float32'), device=DEVICE))
    y_test_t = (torch.tensor(y_test['duration'].to_numpy(dtype='float32'), device=DEVICE),
                torch.tensor(y_test['event'].to_numpy(dtype='float32'), device=DEVICE))
    return X_train_t, y_train_t, X_test_t, y_test_t


# ---------------------- PCA ------------------------
def fit_transform_pca(X_train_np, n_components=N_COMPONENTS):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_np)
    return X_train_pca, pca


# ---------------------- MODEL ----------------------
def create_model(in_features, params, network_class):
    if network_class == Net_3layers:
        net = Net_3layers(in_features, 1, params['hidden1'], params['hidden2'], params['dropout'])
    else:
        net = Net_5layers(in_features, 1, params['hidden1'], params['hidden2'], params['hidden3'],
                          params['hidden4'], params['dropout'])
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(params['lr'])
    return model


# ------------------ CROSS-VALIDATION ------------------
def cross_validate(X, y, fold_indexes, network_class, param_grid, dataset_name):
    #in_features = X.shape[1]
    in_features = N_COMPONENTS  # In caso di PCA
    best_result = {'best_score': -1, 'best_params': None, 'best_model': None}

    total_combs = reduce(operator.mul, (len(v) for v in param_grid.values()))
    print(f"Testing {total_combs } possible parameters combinations")

    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        cindex_scores = []
        brier_scores = []
        times_folds = []
        ibs_scores = []
        model = None
        log = None
        print(f"Testing {i}/{total_combs }".center(100, '='))
        for train_idx, val_idx in fold_indexes:
            X_train_fold_np = X[train_idx].cpu().numpy()
            X_val_fold_np = X[val_idx].cpu().numpy()

            X_train_pca, pca_model = fit_transform_pca(X_train_fold_np)
            X_val_pca = pca_model.transform(X_val_fold_np)

            X_train_fold = torch.tensor(X_train_pca, dtype=torch.float32, device=DEVICE)
            X_val_fold = torch.tensor(X_val_pca, dtype=torch.float32, device=DEVICE)

            # Se non PCA:
            #X_train_fold = torch.tensor(X_train_fold_np, dtype=torch.float32, device=DEVICE)
            #X_val_fold = torch.tensor(X_val_fold_np, dtype=torch.float32, device=DEVICE)

            y_train_fold = (y[0][train_idx], y[1][train_idx])
            y_val_fold = (y[0][val_idx], y[1][val_idx])
            #X_train_fold = X[train_idx]
            #X_val_fold = X[val_idx]

            model = create_model(in_features, params, network_class)
            #if events_val.sum() != 0:
            log = model.fit(X_train_fold, y_train_fold,
                            batch_size=params['batch_size'], epochs=params['epochs'],
                            callbacks=[tt.callbacks.EarlyStopping(patience=25)],
                            verbose=True, val_data=(X_val_fold, y_val_fold))

            _ = model.compute_baseline_hazards()
            surv_df = model.predict_surv_df(X_val_fold)

            # EvalSurv
            ev = EvalSurv(surv_df,
                          y_val_fold[0].cpu().numpy(),
                          y_val_fold[1].cpu().numpy(),
                          censor_surv='km')

            # C-index
            cindex_scores.append(ev.concordance_td())

            # Brier score
            times = surv_df.index.values
            bs = ev.brier_score(times)
            brier_scores.append(list(bs))
            times_folds.append(list(times))

            # Integrated Brier Score
            ibs = ev.integrated_brier_score(times)
            ibs_scores.append(ibs)

        mean_score = np.mean(cindex_scores)
        results.append({'params': params, 'mean_concordance': mean_score, 'std_concordance': np.std(cindex_scores)})
        if mean_score > best_result['best_score']:
            best_result = {'best_score': mean_score, 'best_params': params, 'best_model': model}

        results[-1] |= {
            f"split{i}_c_index": cindex_scores[i]
            for i in range(len(cindex_scores))
        }
        results[-1] |= {
            f"split{i}_brier_score": brier_scores[i]
            for i in range(len(brier_scores))
        }
        results[-1] |= {
            f"split{i}_times": times_folds[i]
            for i in range(len(times_folds))
        }
        results[-1] |= {
            f"split{i}_ibs": ibs_scores[i]
            for i in range(len(ibs_scores))
        }

    results = pd.DataFrame(results)
    network_name = str(network_class).split('.')[1].strip('\'>')
    results.to_csv(os.path.join(ROOT, f'deepsurv_gcv_results\\{network_name}\\{dataset_name}.csv'), index=False)

    print("\n✅ Migliori parametri trovati:")
    print(f"Miglior concordanza: {best_result['best_score']}")
    print(f"Migliori parametri: {best_result['best_params']}")
    print(f"Concordanza media: {results['mean_concordance'].mean()}")
    return best_result, results


# ------------------ SAVE MODEL ------------------
def save_model(final_model, pca_model, subtype, dataset_name, network_name, use_clinical=USE_CLINICAL):
    clinical_tag = "clinical" if use_clinical else "no_clinical"
    model_path = os.path.join(ROOT, 'deepsurv_results', subtype, f"{dataset_name}__{network_name}__{clinical_tag}.pth")
    torch.save(final_model.net.state_dict(), model_path)

    """pca_path = os.path.join(ROOT, 'deepsurv_results', subtype, f"{dataset_name}__{network_name}__{clinical_tag}_pca.pkl")
    import joblib
    joblib.dump(pca_model, pca_path)"""


# ------------------- PLOTS -------------------
def plots(best_res):
    scores = best_res[[col for col in best_res.columns if 'split' in col and 'c_index' in col]].values.flatten()
    times_folds = best_res[[col for col in best_res.columns if 'split' in col and 'times' in col]].values.flatten()
    brier_scores = best_res[[col for col in best_res.columns if 'split' in col and 'brier_score' in col]].values.flatten()
    ibs_folds = best_res[[col for col in best_res.columns if 'split' in col and 'ibs' in col]].values.flatten()

    # Boxplot C-index
    plt.figure(figsize=(6, 5))
    plt.boxplot(scores, vert=True, patch_artist=True)
    plt.title("Distribuzione C-index sui fold")
    plt.ylabel("C-index")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Plot Brier score per fold
    max_common = min(np.max(np.array(t)) for t in times_folds)
    min_common = max(np.min(np.array(t)) for t in times_folds)
    time_grid = np.linspace(min_common, max_common, 300)  # puoi cambiare risoluzione
    brier_interp = []

    for times, bs in zip(times_folds, brier_scores):
        # crea interpolatore lineare
        f = interp1d(times, bs, kind='nearest', bounds_error=False, fill_value=np.nan)
        # valutiamo sul time_grid
        brier_interp.append(f(time_grid))

    # ---- IBS Statistics ----
    ibs_mean = np.mean(ibs_folds)
    ibs_std = np.std(ibs_folds)
    ibs_min = np.min(ibs_folds)
    ibs_max = np.max(ibs_folds)
    ibs_p25 = np.percentile(ibs_folds, 25)
    ibs_p50 = np.percentile(ibs_folds, 50)
    ibs_p75 = np.percentile(ibs_folds, 75)

    plt.figure(figsize=(8, 6))
    for i, bs in enumerate(brier_interp):
        plt.plot(time_grid, bs, alpha=0.6, label=f"Fold {i}")
    plt.title("Brier Score per Fold (ricampionate su griglia comune)\n\n"
              f"IBS — Mean: {ibs_mean:.4f} | Std: {ibs_std:.4f} | "
              f"Min: {ibs_min:.4f} | Max: {ibs_max:.4f} |\n"
              f"P25: {ibs_p25:.4f} | Median: {ibs_p50:.4f} | P75: {ibs_p75:.4f}")
    plt.xlabel("Tempo")
    plt.ylabel("Brier Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def save_plots(final_model, pca_model, X_test_pca_gpu, y_test, dataset_name, network_name):
    X_test_np = X_test_pca_gpu.cpu().numpy()
    durations = y_test['duration'].values
    events = y_test['event'].values

    print(final_model.baseline_hazards_)
    print(final_model.baseline_cumulative_hazards_)

    # Predizione curve di sopravvivenza
    surv = final_model.predict_surv_df(torch.tensor(X_test_np, dtype=torch.float32))

    # Checks
    print("Eventi nel test set:", np.unique(events, return_counts=True))
    print("Durate min/max:", durations.min(), durations.max())
    print("Surv shape:", surv.shape, "Surv nan:", np.isnan(surv.values).sum())

    # EvalSurv
    ev = EvalSurv(surv, durations, events, censor_surv="km")

    time_grid = np.linspace(durations.min(), durations.max(), 100)
    brier_scores = ev.brier_score(time_grid)

    # Integrated Brier Score (IBS)
    ibs = ev.integrated_brier_score(time_grid)
    print(f"📉 IBS (Integrated Brier Score): {ibs:.4f}")

    # ------------ Time-dependent ------------
    cindex_global = ev.concordance_td()
    print(f"📈 C-index (global): {cindex_global:.4f}")

    # -------- PLOT FIGURE -------
    plt.figure(figsize=(18, 14))

    # 1. Survival curves
    plt.subplot(2, 3, 1)
    for i in range(min(30, surv.shape[1])):
        plt.step(surv.index, surv.iloc[:, i], where="post", alpha=0.3)
    plt.title("Survival curves (test set)")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")

    # 2. Training loss
    if hasattr(final_model, "log"):
        log = final_model.log
        plt.subplot(2, 3, 2)
        plt.plot(log.loss, label="train")
        if hasattr(log, "val_loss"):
            plt.plot(log.val_loss, label="valid")
        plt.title("Training / Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    # 3. PCA variance
    plt.subplot(2, 3, 3)
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
    plt.title("PCA cumulative explained variance")
    plt.xlabel("Components")
    plt.ylabel("Cumulative variance")

    # 4. Brier score vs time
    plt.subplot(2, 3, 4)
    plt.plot(time_grid, brier_scores)
    plt.title(f"Brier score over time\nIBS = {ibs:.4f}")
    plt.xlabel("Time")
    plt.ylabel("Brier score")

    # 5. C-index
    plt.subplot(2, 3, 5)
    plt.bar(["C-index"], [cindex_global])
    plt.ylim(0, 1)
    plt.title("Global C-index")

    # 6. IBS (single bar)
    plt.subplot(2, 3, 6)
    plt.bar(["IBS"], [ibs])
    plt.ylim(0, 1)
    plt.title("Integrated Brier Score")

    # ---- Save PNG ----
    fig_path = os.path.join(
        ROOT, "deepsurv_results", f"{dataset_name}__{network_name}.png"
    )
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"📊 Saved plot: {fig_path}")


def main():
    """datasets = [
        'miRNA/clinical_miRNA_normalized_log.csv',
        'miRNA/clinical_miRNA_normalized_quant.csv',
        'mRNA/clinical_mRNA_normalized_log.csv',
        'mRNA/clinical_mRNA_normalized_tpm_log.csv'
    ]"""
    datasets = [
        'miRNA/clinical_miRNA_normalized_log.csv',
    ]

    # Create output dirs
    output_dir = os.path.join(ROOT, 'deepsurv_results')
    os.makedirs(output_dir, exist_ok=True)

    for dataset_file in datasets:
        print("Preparing data...".center(100, '-'))
        dataset_name = os.path.basename(dataset_file).replace(".csv", "")
        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_file))
        subtype = dataset_file.split('/')[0]

        output_dir = os.path.join(ROOT, 'deepsurv_results', subtype)
        os.makedirs(output_dir, exist_ok=True)
        gcv_3_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, 'Net_3layers')
        gcv_5_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, 'Net_5layers')
        os.makedirs(gcv_3_output_dir, exist_ok=True)
        os.makedirs(gcv_5_output_dir, exist_ok=True)

        X, y = prepare_data(dataset)
        X = scale_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                                            stratify=y['event'])
        X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu = data_to_gpu(X_train, y_train, X_test, y_test)

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = list(kfold.split(X_train, y_train['event']))

        # ---------------- 3 LAYERS ----------------
        """param_grid_3 = {
            'hidden1': [32, 64, 128, 256],
            'hidden2': [16, 32, 64, 128],
            'dropout': [0.1, 0.3, 0.5, 0.7],
            'epochs': [150, 300, 500],
            'lr': [1e-2, 1e-3, 1e-4],
            'batch_size': [32, 64]
        }"""
        param_grid_3 = {
            'hidden1': [32],
            'hidden2': [128],
            'dropout': [0.5],
            'epochs': [150],
            'lr': [1e-2, 1e-3, 1e-4],
            'batch_size': [64]
        }

        best_3, results_3 = cross_validate(X_train_gpu, y_train_gpu, fold_indexes, Net_3layers, param_grid_3, dataset_name)
        best_res_3 = results_3[results_3['params'] == best_3['best_params']]
        plots(best_res_3)

        # ---------------- 5 LAYERS ----------------
        """param_grid_5 = {
            'hidden1': [128, 256], 'hidden2': [64, 128], 'hidden3': [32, 64], 'hidden4': [16, 32],
            'dropout': [0.3, 0.5], 'lr': [0.01, 0.001, 0.0001], 'batch_size': [32, 64, 128],
            'epochs': [200, 500], 'weight_decay': [1e-6, 1e-5, 1e-4], 'lr_factor': [0.7, 0.5]
        }"""
        param_grid_5 = {
            'hidden1': [128, 256], 'hidden2': [64, 128], 'hidden3': [32], 'hidden4': [16],
            'dropout': [0.3], 'lr': [0.01], 'batch_size': [32],
            'epochs': [200], 'weight_decay': [1e-6], 'lr_factor': [0.7]
        }
        best_5, results_5 = cross_validate(X_train_gpu, y_train_gpu, fold_indexes, Net_5layers, param_grid_5,
                                           dataset_name)
        best_res_5 = results_5[results_5['params'] == best_5['best_params']]
        plots(best_res_5)


if __name__ == "__main__":
    main()
