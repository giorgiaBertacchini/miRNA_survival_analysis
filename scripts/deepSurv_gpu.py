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


def data_to_gpu(X, y):
    if hasattr(X, "values"):
        X_np = X.values
    else:
        X_np = X
    X_t = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    y_t = (torch.tensor(y['duration'].to_numpy(dtype='float32'), device=DEVICE),
           torch.tensor(y['event'].to_numpy(dtype='float32'), device=DEVICE))
    return X_t, y_t


# ---------------------- PCA ------------------------
def fit_transform_pca(X_df, n_components=N_COMPONENTS):
    """
    Applica PCA solo alle colonne non cliniche (hsa*, gene.*)
    e restituisce una matrice completa X_pca + colonne cliniche originali.
    """
    pca_cols = [col for col in X_df.columns if col.startswith('hsa') or col.startswith('gene.')]
    clinical_cols = [col for col in X_df.columns if col not in pca_cols]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_df[pca_cols].values)

    X_combined = np.concatenate([X_df[clinical_cols].values, X_pca], axis=1)

    new_feature_names = (
        [f"PCA_{i+1}" for i in range(n_components)] +
        clinical_cols
    )

    return X_combined, pca, new_feature_names


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
def cross_validate(X, y, fold_indexes, subtype, network_class, param_grid, dataset_name):
    in_features = X.shape[1]
    best_result = {'best_score': -1, 'best_params': None, 'best_model': None}

    total_combs = reduce(operator.mul, (len(v) for v in param_grid.values()))
    print(f"Testing {total_combs } possible parameters combinations")

    results = []
    saved_times_csv = False

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        cindex_scores = []
        brier_scores = []
        times_folds = []
        ibs_scores = []
        model = None

        print(f"Testing {i}/{total_combs }".center(100, '='))
        for train_idx, val_idx in fold_indexes:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            #X_train_fold_np = X[train_idx].cpu().numpy()
            #X_val_fold_np = X[val_idx].cpu().numpy()

            #X_train_pca, pca_model = fit_transform_pca(X_train_fold_np)
            #X_val_pca = pca_model.transform(X_val_fold_np)

            #X_train_fold = torch.tensor(X_train_pca, dtype=torch.float32, device=DEVICE)
            #X_val_fold = torch.tensor(X_val_pca, dtype=torch.float32, device=DEVICE)

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

        if not saved_times_csv:
            times_dict = {f"times_fold{i + 1}": [times_folds[i]] for i in range(len(times_folds))}
            df_times = pd.DataFrame(times_dict)
            save_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            df_times.to_csv(os.path.join(save_dir, "times_by_fold.csv"), index=False)

            saved_times_csv = True

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
            f"split{i}_ibs": ibs_scores[i]
            for i in range(len(ibs_scores))
        }

    results = pd.DataFrame(results)
    network_name = str(network_class).split('.')[1].strip('\'>')
    results.to_csv(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, f'cv_results_{network_name}.csv'), index=False)

    print("\n✅ Migliori parametri trovati:")
    print(f"Miglior concordanza: {best_result['best_score']}")
    print(f"Migliori parametri: {best_result['best_params']}")
    print(f"Concordanza media: {results['mean_concordance'].mean()}")
    return best_result, results


# ------------------ SAVE MODEL ------------------
def save_model(final_model, subtype, dataset_name, network_name, use_clinical=USE_CLINICAL):
    clinical_tag = "clinical" if use_clinical else "no_clinical"
    model_path = os.path.join(ROOT, 'deepsurv_results', subtype, f"{dataset_name}__{network_name}__{clinical_tag}.pth")
    torch.save(final_model.net.state_dict(), model_path)


# ------------------- PLOTS -------------------
def plots(best_res_3, best_res_5, pca_model, times_csv_path, feature_names, plot_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    ax1, ax_empty, ax2, ax3, ax4, ax5 = axes.flatten()

    ax_empty.axis("off")
    ax_empty.set_title("")

    # -----------------------------------------------------------
    # (A) PCA FEATURE IMPORTANCE
    # -----------------------------------------------------------
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    pc1_importances = np.abs(loadings[:, 0])

    top_k = 20
    idx_top = np.argsort(pc1_importances)[-top_k:][::-1]
    top_features = np.array(feature_names)[idx_top]
    top_values = pc1_importances[idx_top]

    ax1.barh(top_features, top_values)
    ax1.set_title("Top PCA Feature Importances (PC1)")
    ax1.set_xlabel("Loading |PC1|")
    ax1.invert_yaxis()

    # -----------------------------------------------------------
    # CARICAMENTO TIMES UNA SOLA VOLTA
    # -----------------------------------------------------------
    df_times = pd.read_csv(times_csv_path)
    times_folds = []
    for col in df_times.columns:
        str_list = df_times[col].iloc[0]
        cleaned_list = [
            float(s.strip().replace("np.float32(", "").replace(")", ""))
            for s in str_list.strip("[]").split(",")
        ]
        times_folds.append(cleaned_list)

    # -----------------------------------------------------------
    # (B) MODELLO 3-LAYER → BOX C-INDEX
    # -----------------------------------------------------------
    scores_3 = best_res_3[[col for col in best_res_3.columns if 'split' in col and 'c_index' in col]].values.flatten()

    ax2.boxplot(scores_3, vert=True, patch_artist=True)
    ax2.set_title("Distribuzione C-index sui fold")
    ax2.set_ylabel("C-index")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # (C) MODELLO 3-LAYER → BRIER CURVES
    # -----------------------------------------------------------
    brier_scores_3 = best_res_3[
        [col for col in best_res_3.columns if 'split' in col and 'brier_score' in col]].values.flatten()
    ibs_folds_3 = best_res_3[[col for col in best_res_3.columns if 'split' in col and 'ibs' in col]].values.flatten()

    max_common = min(np.max(np.array(t)) for t in times_folds)
    min_common = max(np.min(np.array(t)) for t in times_folds)
    time_grid = np.linspace(min_common, max_common, 300)  # puoi cambiare risoluzione

    ibs_mean = np.mean(ibs_folds_3)
    ibs_std = np.std(ibs_folds_3)
    ibs_min = np.min(ibs_folds_3)
    ibs_max = np.max(ibs_folds_3)
    ibs_p25 = np.percentile(ibs_folds_3, 25)
    ibs_p50 = np.percentile(ibs_folds_3, 50)
    ibs_p75 = np.percentile(ibs_folds_3, 75)

    for i, (times, bs) in enumerate(zip(times_folds, brier_scores_3)):
        f = interp1d(times, bs, kind='nearest', bounds_error=False, fill_value=np.nan)
        ax3.plot(time_grid, f(time_grid), alpha=0.6, label=f"Fold {i}")

    ax3.set_title("Brier Score per Fold (ricampionate su griglia comune)\n\n"
              f"IBS — Mean: {ibs_mean:.4f} | Std: {ibs_std:.4f} | "
              f"Min: {ibs_min:.4f} | Max: {ibs_max:.4f} |\n"
              f"P25: {ibs_p25:.4f} | Median: {ibs_p50:.4f} | P75: {ibs_p75:.4f}")
    ax3.set_xlabel("Tempo")
    ax3.set_ylabel("Brier Score")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    # -----------------------------------------------------------
    # (D) MODELLO 5-LAYER → BOX C-INDEX
    # -----------------------------------------------------------
    scores_5 = best_res_5[[col for col in best_res_5.columns if 'split' in col and 'c_index' in col]].values.flatten()

    ax4.boxplot(scores_5, vert=True, patch_artist=True)
    ax4.set_title("Distribuzione C-index sui fold")
    ax4.set_ylabel("C-index")
    ax4.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # (E) MODELLO 5-LAYER → BRIER CURVES
    # -----------------------------------------------------------
    brier_scores_5 = best_res_5[
        [col for col in best_res_5.columns if 'split' in col and 'brier_score' in col]].values.flatten()
    ibs_folds_5 = best_res_5[[col for col in best_res_5.columns if 'split' in col and 'ibs' in col]].values.flatten()

    max_common = min(np.max(np.array(t)) for t in times_folds)
    min_common = max(np.min(np.array(t)) for t in times_folds)
    time_grid = np.linspace(min_common, max_common, 300)  # puoi cambiare risoluzione
    brier_interp = []

    # ---- IBS Statistics ----
    ibs_mean = np.mean(ibs_folds_5)
    ibs_std = np.std(ibs_folds_5)
    ibs_min = np.min(ibs_folds_5)
    ibs_max = np.max(ibs_folds_5)
    ibs_p25 = np.percentile(ibs_folds_5, 25)
    ibs_p50 = np.percentile(ibs_folds_5, 50)
    ibs_p75 = np.percentile(ibs_folds_5, 75)

    for i, (times, bs) in enumerate(zip(times_folds, brier_scores_5)):
        f = interp1d(times, bs, kind='nearest', bounds_error=False, fill_value=np.nan)
        ax5.plot(time_grid, f(time_grid), alpha=0.6, label=f"Fold {i}")

    ax5.set_title("Brier Score per Fold (ricampionate su griglia comune)\n\n"
                  f"IBS — Mean: {ibs_mean:.4f} | Std: {ibs_std:.4f} | "
                  f"Min: {ibs_min:.4f} | Max: {ibs_max:.4f} |\n"
                  f"P25: {ibs_p25:.4f} | Median: {ibs_p50:.4f} | P75: {ibs_p75:.4f}")
    ax5.set_xlabel("Tempo")
    ax5.set_ylabel("Brier Score")
    ax5.legend()
    ax5.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
    fig.savefig(plot_path)
    plt.close()
    print(f"📊 Saved plot: {plot_path}")


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
        gcv_3_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name)
        gcv_5_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name)
        os.makedirs(gcv_3_output_dir, exist_ok=True)
        os.makedirs(gcv_5_output_dir, exist_ok=True)

        X, y = prepare_data(dataset)
        X = scale_data(X)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED,
        #                                                    stratify=y['event'])

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = list(kfold.split(X, y['event']))

        X_pca, pca_model, new_feature_names = fit_transform_pca(X)
        X_gpu, y_gpu = data_to_gpu(X_pca, y)
        feature_names = X.columns

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
        best_3, results_3 = cross_validate(X_gpu, y_gpu, fold_indexes, subtype, Net_3layers, param_grid_3, dataset_name)
        best_res_3 = results_3[results_3['params'] == best_3['best_params']]
        times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
        save_model(best_3['best_model'], subtype, dataset_name, '3layers')

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
        best_5, results_5 = cross_validate(X_gpu, y_gpu, fold_indexes, subtype, Net_5layers, param_grid_5,
                                           dataset_name)
        best_res_5 = results_5[results_5['params'] == best_5['best_params']]
        save_model(best_3['best_model'], subtype, dataset_name, '3layers')

        plot_path = os.path.join(ROOT, 'deepsurv_results', subtype, f"{dataset_name}__summary.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plots(best_res_3, best_res_5, pca_model, times_csv_path, feature_names, plot_path)


if __name__ == "__main__":
    main()
