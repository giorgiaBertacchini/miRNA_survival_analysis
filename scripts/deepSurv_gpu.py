import random
import os
from functools import reduce
from scipy.interpolate import interp1d

import numpy as np
import operator
import pandas as pd
import torch
import shap
import json
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

"""
DeepSurv pipeline with PCA-based dimensionality reduction,
cross-validated hyperparameter search, and SHAP-based model interpretability.
"""

# ---------------------- CONFIGURATION ----------------------
NUM_FOLDS = 5  # Number of CV folds TODO aumentare?
SEED = 42  # Random seed for reproducibility
USE_CLINICAL = True  # Whether to include clinical variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_COMPONENTS = 50  # Number of PCA components (increase for mRNA) TODO aumentare?

print(f"Running on device: {DEVICE}")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)

# Paths
base = os.path.basename(os.getcwd())
list_path = os.getcwd().split(os.sep)
list_path.pop()
# list.pop(list.index(base))
ROOT = '\\'.join(list_path)
#ROOT = os.path.dirname(os.getcwd()) + "/mbulgarelli"
DATA_PATH = os.path.join(ROOT, 'datasets/preprocessed')


# ---------------------- DATA -----------------------
def prepare_data(dataset, use_clinical=USE_CLINICAL):
    """
    Splits the dataset into feature matrix X and survival targets y.
    Optionally includes or excludes clinical variables.
    """
    y_cols = ['Death', 'days_to_last_followup', 'days_to_death']
    y = dataset[['Death', 'days_to_last_followup']].copy()
    y = y.rename(columns={'Death': 'event', 'days_to_last_followup': 'duration'})

    if use_clinical:
        print("Including clinical variables in the feature set.")
        X_cols = [col for col in dataset.columns if col not in y_cols]
    else:
        print("Excluding clinical variables from the feature set.")
        miRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'hsa' not in col]
        mRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'gene.' not in col]
        X_cols = [col for col in dataset.columns if
                  col not in y_cols and not col in miRNA_clinical_cols and col not in mRNA_clinical_cols]

    X = dataset[X_cols].copy()
    return X, y


def scale_data(X):
    """
    Applies standard scaling to molecular features (miRNA/mRNA)
    and age at diagnosis.
    """
    scaler = StandardScaler()
    gene_cols = [col for col in X.columns if 'hsa' in col or 'gene.' in col]
    gene_cols.append('age_at_initial_pathologic_diagnosis')
    X[gene_cols] = scaler.fit_transform(X[gene_cols])
    return X


def data_to_gpu(X, y):
    """
    Converts feature matrix and survival targets to PyTorch tensors
    and moves them to the selected device (CPU/GPU).
    """
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
    Performs PCA on non-clinical features (hsa*, gene.*) only,
    and returns a combined feature matrix including the PCA
    components and the original clinical variables.
    """
    print(f"Applying PCA with {n_components} components to molecular features.")
    pca_cols = [col for col in X_df.columns if col.startswith('hsa') or col.startswith('gene.')]
    clinical_cols = [col for col in X_df.columns if col not in pca_cols]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_df[pca_cols].values)

    X_combined = np.concatenate([X_df[clinical_cols].values, X_pca], axis=1)

    new_feature_names = (
            [f"PCA_{i + 1}" for i in range(n_components)] +
            clinical_cols
    )

    return X_combined, pca, new_feature_names


def transform_pca(X_df, model_pca):
    pca_cols = [col for col in X_df.columns if col.startswith('hsa') or col.startswith('gene.')]
    clinical_cols = [col for col in X_df.columns if col not in pca_cols]

    X_pca = model_pca.transform(X_df[pca_cols].values)

    X_combined = np.concatenate([X_df[clinical_cols].values, X_pca], axis=1)

    return X_combined


# ---------------------- MODEL ----------------------
def create_model(in_features, params, network_class):
    """
    Instantiates a DeepSurv model (CoxPH) with the specified
    neural network architecture and hyperparameters.
    """
    if network_class == Net_3layers:
        net = Net_3layers(in_features, 1, params['hidden1'], params['hidden2'], params['dropout'])
    else:
        net = Net_5layers(in_features, 1, params['hidden1'], params['hidden2'], params['hidden3'],
                          params['hidden4'], params['dropout'])
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(params['lr'])
    return model


# ------------------ CROSS-VALIDATION ------------------
def grid_searches(X, y, fold_indexes, subtype, network_class, param_grid, dataset_name):
    print(f"\nStarting grid search for {network_class.__name__}")

    in_features = X.shape[1]
    best_result = {'best_score': -1, 'best_params': None, 'best_model': None}

    total_combs = reduce(operator.mul, (len(v) for v in param_grid.values()))

    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        cindex_scores = []

        print(f"Testing {i}/{total_combs}".center(100, '='))
        print(f"Parameters: {params}")
        for train_idx, val_idx in fold_indexes:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = (y[0][train_idx], y[1][train_idx])
            y_val_fold = (y[0][val_idx], y[1][val_idx])

            model = create_model(in_features, params, network_class)
            model.fit(X_train_fold, y_train_fold,
                      batch_size=params['batch_size'], epochs=params['epochs'],
                      callbacks=[tt.callbacks.EarlyStopping(patience=25)],
                      verbose=True, val_data=(X_val_fold, y_val_fold))

            _ = model.compute_baseline_hazards()
            surv_df = model.predict_surv_df(X_val_fold)

            ev = EvalSurv(surv_df,
                          y_val_fold[0].cpu().numpy(),
                          y_val_fold[1].cpu().numpy(),
                          censor_surv='km')

            # C-index
            cindex_scores.append(ev.concordance_td())

        mean_score = np.mean(cindex_scores)

        fold_result = {
            'params': params,
            'mean_concordance': mean_score,
            'std_concordance': np.std(cindex_scores)
        }
        for f in range(len(cindex_scores)):
            fold_result[f"split{f}_c_index"] = cindex_scores[f]
        results.append(fold_result)

        if mean_score > best_result['best_score']:
            best_result = {'best_score': mean_score, 'best_params': params}

    print("\nBest hyperparameters identified:")
    print(f"Best concordance index: {best_result['best_score']:.4f}")
    print(f"Optimal parameter set: {best_result['best_params']}")

    net_name = network_class.__name__
    results = pd.DataFrame(results)
    results.to_csv(
        os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, f'gcv_results_{net_name}.csv'),
        index=False)

    best_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name,
                             f'gcv_best_results_{net_name}.json')
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=4)

    return best_result, results


def cross_validate(X, y, fold_indexes, params, network_class, subtype, dataset_name):
    print(f"\nRunning cross-validation for {network_class.__name__}")
    in_features = int(X.shape[1])
    cindex_scores = []
    brier_scores = []
    times_folds = []
    ibs_scores = []
    models = []

    to_int = {"batch_size", "epochs", "hidden1", "hidden2", "hidden3", "hidden4"}
    params = {
        k: int(v) if k in to_int else v
        for k, v in params.items()
    }

    for train_idx, val_idx in fold_indexes:
        print(f"\nTraining fold {len(cindex_scores) + 1}/{len(fold_indexes)}")
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = (y[0][train_idx], y[1][train_idx])
        y_val_fold = (y[0][val_idx], y[1][val_idx])

        model = create_model(in_features, params, network_class)
        log = model.fit(X_train_fold, y_train_fold,
                        batch_size=params['batch_size'], epochs=params['epochs'],
                        callbacks=[tt.callbacks.EarlyStopping(patience=25)],
                        verbose=True)

        _ = model.compute_baseline_hazards()
        surv_df = model.predict_surv_df(X_val_fold)

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

        # Save model
        models.append(model)

    # Save times
    times_dict = {f"times_fold{i + 1}": [times_folds[i]] for i in range(len(times_folds))}
    df_times = pd.DataFrame(times_dict)

    rows = []
    for f in range(len(cindex_scores)):
        rows.append({
            "split": f,
            "c_index": cindex_scores[f],
            "brier_score": list(brier_scores[f]),
            "ibs": ibs_scores[f]
        })

    # Save results
    fold_result = pd.DataFrame(rows)
    fold_result.to_csv(
        os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name,
                     f'cv_results_{network_class.__name__}.csv'),
        index=False)

    df_times.to_csv(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv"),
                    index=False)

    for i, model in enumerate(models):
        save_model(model, subtype, dataset_name, network_name=network_class.__name__)

    return models, fold_result, df_times


# ------------------ SAVE MODEL ------------------
def save_model(final_model, subtype, dataset_name, network_name, use_clinical=USE_CLINICAL):
    clinical_tag = "clinical" if use_clinical else "no_clinical"

    dict_path = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
    os.makedirs(dict_path, exist_ok=True)
    model_path = os.path.join(dict_path, f"{network_name}__{clinical_tag}.pth")

    torch.save(final_model.net.state_dict(), model_path)


# ------- MODEL INTERPRETABILITY (SHAP) -----------
class WrappedNet(torch.nn.Module):
    def __init__(self, net, device):
        super().__init__()
        self.net = net
        self.device = device

    def forward(self, x):
        # Adds an extra output dimension required by SHAP
        x = x.to(self.device)
        return self.net(x).unsqueeze(1)


def explanation(ax, models, X_gpu, feature_names, file_path):
    print("Computing SHAP values for model ensemble...")
    all_shap_values = []
    X = X_gpu.detach().float().to(DEVICE)

    weights = [1.0 / len(models)] * len(models)

    for m in models:
        net = WrappedNet(m.net, DEVICE).to(DEVICE)
        net.eval()

        bg_idx = np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)
        background = X[bg_idx]

        # Explainer SHAP
        explainer = shap.GradientExplainer(net, background)
        shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values[:, :, 0]

        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()

        all_shap_values.append(shap_values)

    """# Summary plot
    rng = np.random.default_rng(seed=42)
    shap.summary_plot(shap_values, features=X_cpu.numpy(), feature_names=feature_names, show=False, rng=rng)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()"""

    # Store all SHAP values and compute their mean across models
    all_shap_values = np.array(all_shap_values)
    n_models, n_samples, n_features = all_shap_values.shape

    rows = []
    for m_idx in range(n_models):
        for s_idx in range(n_samples):
            for f_idx in range(n_features):
                rows.append({
                    "model_id": m_idx,
                    "sample_id": s_idx,
                    "feature": feature_names[f_idx],
                    "shap_value": all_shap_values[m_idx, s_idx, f_idx]
                })
    df_shap = pd.DataFrame(rows)
    df_shap.to_csv(file_path, index=False)
    print(f"SHAP values saved to: {file_path}")

    # Convertiamo in array e calcoliamo media pesata
    all_shap_values = np.array(all_shap_values)  # shape: (n_models, n_samples, n_features)
    mean_shap_across_models = np.tensordot(weights, all_shap_values, axes=(0, 0))  # shape: (n_samples, n_features)

    # Calcola importanza media assoluta
    feature_names = list(feature_names)
    # mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_abs_shap = np.mean(np.abs(mean_shap_across_models), axis=0)

    # Ordina feature per importanza
    top_idx = np.argsort(mean_abs_shap)[::-1][:35]
    top_values = mean_abs_shap[top_idx]
    top_features = [str(feature_names[i]) for i in top_idx]

    # Plot
    ax.barh(top_features[::-1], top_values[::-1], color='skyblue')
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top Feature Importances (SHAP) - DeepSurv")
    ax.invert_yaxis()


# ------------------- PLOTS -------------------
def plots(models_3, best_res_3, models_5, best_res_5, pca_model, X, times_csv_path, feature_names, results_dir):
    # -----------------------------------------------------------
    # (A) PCA FEATURE IMPORTANCE
    # -----------------------------------------------------------
    print("Plotting PCA feature importances.")

    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    pc1_importances = np.abs(loadings[:, 0])

    top_k = 25
    idx_top = np.argsort(pc1_importances)[-top_k:][::-1]
    top_features = np.array(feature_names)[idx_top]
    top_values = pc1_importances[idx_top]

    figPCA, ax = plt.subplots(figsize=(8, 7))

    ax.barh(top_features, top_values)
    ax.set_title("Top PCA Feature Importances")
    ax.set_xlabel("Loading |PC1|")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
    figPCA.savefig(results_dir + "/pca_feature_importances.png")
    plt.close()
    print(f"Saved plot: {results_dir}/pca_feature_importances.png")

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
    # (B) MODELLO 3-LAYER
    # -----------------------------------------------------------
    print("\nPlotting DeepSurv 3-layer model results.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax1, ax2, ax_explain_3 = axes.flatten()
    fig.suptitle("Network 3 layers", fontsize=14, fontweight='bold')

    # -----------------------------------------------------------
    # (B.1) MODELLO 3-LAYER → BOX C-INDEX
    # -----------------------------------------------------------
    scores_3 = best_res_3["c_index"].values

    ax1.boxplot(scores_3, vert=True, patch_artist=True)
    ax1.set_title("C-index distribution across folds")
    ax1.set_ylabel("C-index")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # (B.2) MODELLO 3-LAYER → BRIER CURVES
    # -----------------------------------------------------------
    brier_scores_3 = best_res_3["brier_score"].values
    ibs_folds_3 = best_res_3["ibs"].values

    max_common = min(np.max(np.array(t)) for t in times_folds)
    min_common = max(np.min(np.array(t)) for t in times_folds)
    time_grid = np.linspace(min_common, max_common, 300)

    ibs_mean = np.mean(ibs_folds_3)
    ibs_std = np.std(ibs_folds_3)
    ibs_min = np.min(ibs_folds_3)
    ibs_max = np.max(ibs_folds_3)
    ibs_p25 = np.percentile(ibs_folds_3, 25)
    ibs_p50 = np.percentile(ibs_folds_3, 50)
    ibs_p75 = np.percentile(ibs_folds_3, 75)

    for i, (times, bs) in enumerate(zip(times_folds, brier_scores_3)):
        f = interp1d(times, bs, kind='nearest', bounds_error=False, fill_value=np.nan)
        ax2.plot(time_grid, f(time_grid), alpha=0.6, label=f"Fold {i}")

    ax2.set_title("Time-Dependent Brier Score across folds")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Brier Score")
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2.text(
        0.02, 0.98,
        f"IBS\n"
        f"mean = {ibs_mean:.4f}\n"
        f"std  = {ibs_std:.4f}\n"
        f"min  = {ibs_min:.4f}\n"
        f"max  = {ibs_max:.4f}\n"
        f"P25 = {ibs_p25:.4f}\n"
        f"P50 = {ibs_p50:.4f}\n"
        f"P75 = {ibs_p75:.4f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # -----------------------------------------------------------
    # (B.3) MODELLO 3-LAYER → SHAP values
    # -----------------------------------------------------------
    explanation(ax_explain_3, models_3, X, feature_names, results_dir + "/shap_values_3layers.csv")

    plt.tight_layout(pad=1.0)
    plt.show()
    fig.savefig(results_dir + "/results_3layers.png")
    plt.close()
    print(f"Saved plot: {results_dir}/results_3layers.png")

    # -----------------------------------------------------------
    # (C) MODELLO 5-LAYER
    # -----------------------------------------------------------
    print("\nPlotting DeepSurv 5-layer model results.")

    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax4, ax5, ax_explain_5 = axes2.flatten()
    fig2.suptitle("Network 5 layers", fontsize=14, fontweight='bold')

    # -----------------------------------------------------------
    # (C.1) MODELLO 5-LAYER → BOX C-INDEX
    # -----------------------------------------------------------
    scores_5 = best_res_5["c_index"].values

    ax4.boxplot(scores_5, vert=True, patch_artist=True)
    ax4.set_title("C-index distribution across folds")
    ax4.set_ylabel("C-index")
    ax4.set_xticks([])
    ax4.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # (C.2) MODELLO 5-LAYER → BRIER CURVES
    # -----------------------------------------------------------
    brier_scores_5 = best_res_5["brier_score"].values
    ibs_folds_5 = best_res_5["ibs"].values

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

    ax5.set_title("Time-Dependent Brier Score across folds")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Brier Score")
    ax5.legend(loc='lower right')
    ax5.grid(True, linestyle="--", alpha=0.4)

    ax5.text(
        0.02, 0.98,
        f"IBS\n"
        f"mean = {ibs_mean:.4f}\n"
        f"std  = {ibs_std:.4f}\n"
        f"min  = {ibs_min:.4f}\n"
        f"max  = {ibs_max:.4f}\n"
        f"P25 = {ibs_p25:.4f}\n"
        f"P50 = {ibs_p50:.4f}\n"
        f"P75 = {ibs_p75:.4f}",
        transform=ax5.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # -----------------------------------------------------------
    # (C.3) MODELLO 5-LAYER → SHAP values
    # -----------------------------------------------------------
    explanation(ax_explain_5, models_5, X, feature_names, results_dir + "/shap_values_5layers.csv")

    plt.tight_layout(pad=1.0)
    plt.show()
    fig2.savefig(results_dir + "/results_5layers.png")
    plt.close()
    print(f"Saved plot: {results_dir}/results_5layers.png")


def main():
    print("=" * 100)
    print("Starting DeepSurv pipeline")
    print("=" * 100)

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

        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Subtype: {subtype}")

        output_dir = os.path.join(ROOT, 'deepsurv_results', subtype)
        os.makedirs(output_dir, exist_ok=True)
        gcv_3_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name)
        gcv_5_output_dir = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name)
        os.makedirs(gcv_3_output_dir, exist_ok=True)
        os.makedirs(gcv_5_output_dir, exist_ok=True)

        X, y = prepare_data(dataset)
        X = scale_data(X)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y['event'])

        X_pca, pca_model, new_feature_names = fit_transform_pca(X)
        X_pca_gpu, y_pca_gpu = data_to_gpu(X_pca, y)
        feature_names = X.columns
        # X_pca, pca_model, new_feature_names = fit_transform_pca(X_train)
        # X_gpu, y_gpu = data_to_gpu(X_pca, y_train)
        # feature_names = new_feature_names

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = list(kfold.split(X, y['event']))
        # fold_indexes = list(kfold.split(X_train, y_train['event']))

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
        print("\nGrid search for best params...")
        gcv_best_3, gcv_results_3 = grid_searches(X_pca_gpu, y_pca_gpu, fold_indexes, subtype, Net_3layers,
                                                  param_grid_3, dataset_name)
        # best_res_3 = gcv_results_3[gcv_results_3['params'] == gcv_best_3['best_params']]

        print("\nCross validation on best params...")
        models_3, cv_results_3, df_times = cross_validate(X_pca_gpu, y_pca_gpu, fold_indexes, gcv_best_3['best_params'],
                                                          Net_3layers, subtype, dataset_name)

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

        print("\nGrid search for best params...")
        gcv_best_5, gcv_results_5 = grid_searches(X_pca_gpu, y_pca_gpu, fold_indexes, subtype, Net_5layers,
                                                  param_grid_5, dataset_name)

        print("\nCross validation on best params...")
        models_5, cv_results_5, _ = cross_validate(X_pca_gpu, y_pca_gpu, fold_indexes, gcv_best_5['best_params'],
                                                   Net_5layers, subtype, dataset_name)

        # ---------------- PLOTS ----------------
        results_dir = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
        os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
        plots(models_3, cv_results_3, models_5, cv_results_5, pca_model, X_pca_gpu, times_csv_path, feature_names,
              results_dir)

        print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
