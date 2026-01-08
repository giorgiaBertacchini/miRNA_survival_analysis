"""
DeepSurv pipeline with PCA-based dimensionality reduction,
cross-validated hyperparameter search, and SHAP-based model interpretability.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import interp1d
import scipy.integrate

# Network custom architectures
from networks import Net_3layers, Net_5layers

# Utils
from deepSurv_utils import reproducibility, path_config, prepare_data, scale_data, data_to_gpu, create_model
from deepSurv_utils import DecayLR, LrLogger, grid_searches, cross_validate


# Fix for scipy version compatibility
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson


# ---------------------- CONFIGURATION ----------------------
NUM_FOLDS = 9  # Number of CV folds
SEED = 42  # Random seed for reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_COMPONENTS = 50 #50  # Number of PCA components (increase for mRNA -> 200)
GENE_STARTS_WITH = ("hsa", "gene.")  # Prefixes for gene/miRNA columns

print(f"Running on device: {DEVICE}\n")

# ---------------------- REPRODUCIBILITY ----------------------
reproducibility(SEED)

# ---------------------- PATHS ----------------------
ROOT, DATA_PATH = path_config()


# ---------------------- PCA ------------------------
def fit_transform_pca(X_df, n_components=N_COMPONENTS):
    """
    Performs PCA on non-clinical features (hsa*, gene.*) only,
    and returns a combined feature matrix including the PCA
    components and the original clinical variables.
    """
    pca_cols = [col for col in X_df.columns if col.startswith('hsa') or col.startswith('gene.')]
    clinical_cols = [col for col in X_df.columns if col not in pca_cols]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_df[pca_cols].values)

    X_combined = np.concatenate([X_df[clinical_cols].values, X_pca], axis=1)
    new_feature_names = (
            clinical_cols +
            [f"PCA_{i + 1}" for i in range(n_components)]
    )

    return X_combined, pca, new_feature_names


def transform_pca(X_df, model_pca):
    pca_cols = [col for col in X_df.columns if col.startswith('hsa') or col.startswith('gene.')]
    clinical_cols = [col for col in X_df.columns if col not in pca_cols]

    X_pca = model_pca.transform(X_df[pca_cols].values)
    X_combined = np.concatenate([X_df[clinical_cols].values, X_pca], axis=1)
    return X_combined


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


def explanation_pca_to_gene(ax, models, X_pca_gpu, pca, gene_cols, clinical_cols, file_path, top_k=30):
    """
    SHAP calculation in PCA space and projection to gene space.
    Saves the mean absolute SHAP values per gene to a CSV file and plots the top_k
    """
    print("Computing SHAP values (PCA → gene space)...")

    X = X_pca_gpu.detach().float().to(DEVICE)
    all_shap_projected = []

    for model in models:
        net = WrappedNet(model.net, DEVICE).to(DEVICE)
        net.eval()

        bg_idx = np.random.choice(X.shape[0], min(300, X.shape[0]), replace=False)
        background = X[bg_idx]

        explainer = shap.GradientExplainer(net, background)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values[:, :, 0]

        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()

        # ---------------- split ----------------
        n_clinical = len(clinical_cols)
        shap_clinical = shap_values[:, :n_clinical]
        shap_pca = shap_values[:, n_clinical:]

        # ---------------- PCA → gene ----------------
        shap_gene = shap_pca @ pca.components_  # (samples, genes)

        shap_full = np.concatenate([shap_clinical, shap_gene], axis=1)
        all_shap_projected.append(shap_full)

        del net
        del explainer
        torch.cuda.empty_cache()

    all_shap_projected = np.array(all_shap_projected)
    mean_abs_shap = np.mean(np.abs(all_shap_projected), axis=(0, 1))

    feature_names = clinical_cols + gene_cols

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    df.to_csv(file_path, index=False)
    print(f"SHAP gene-level values saved to: {file_path}")

    # ---------------- plot ----------------
    top = df.head(top_k)[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")
    ax.set_title("Top Feature Importances (SHAP, gene-level)")
    ax.set_xlabel("Mean |SHAP value|")


# ------------------- PCA PLOT -------------------
def pca_plot(pca, gene_cols, results_dir):
    # PCA FEATURE IMPORTANCE
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc1_importances = np.abs(loadings[:, 0])

    pca_feature_names = np.array(gene_cols)
    if pc1_importances.shape[0] != pca_feature_names.shape[0]:
        min_len = min(pc1_importances.shape[0], pca_feature_names.shape[0])
        pc1_importances = pc1_importances[:min_len]
        pca_feature_names = pca_feature_names[:min_len]

    # Top 25 features
    top_k = 25
    idx_top = np.argsort(pc1_importances)[-top_k:][::-1]
    top_features = pca_feature_names[idx_top]
    top_values = pc1_importances[idx_top]

    figPCA, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_features, top_values, color='skyblue')
    ax.set_title("Genes contributing to PC1 variance (unsupervised, not model importance)")
    ax.set_xlabel("|PC1 loading|")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    figPCA.savefig(results_dir + "/pca_feature_importances.png")
    plt.close()


# ------------------- PLOTS -------------------
def plots(models_n, best_res_n, pca, X_pca_gpu, times_csv_path, gene_cols, clinical_cols, results_dir, n=3):
    # LOAD TIMES
    df_times = pd.read_csv(times_csv_path)
    times_folds = []
    for col in df_times.columns:
        str_list = df_times[col].iloc[0]
        cleaned_list = [
            float(s.strip().replace("np.float32(", "").replace(")", ""))
            for s in str_list.strip("[]").split(",")
        ]
        times_folds.append(cleaned_list)

    print(f"\nPlotting DeepSurv {n}-layer model results.")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax1, ax2, ax_explain = axes.flatten()
    fig.suptitle(f"Network {n} layers", fontsize=14, fontweight='bold')

    # -----------------------------------------------------------
    # MODEL n-LAYER → BOX C-INDEX
    # -----------------------------------------------------------
    scores_n = best_res_n["c_index"].values

    ax1.boxplot(scores_n, vert=True, patch_artist=True)
    ax1.set_title("C-index distribution across folds")
    ax1.set_ylabel("C-index")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # MODEL n-LAYER → BRIER CURVES
    # -----------------------------------------------------------
    brier_scores_n = best_res_n["brier_score"].values
    ibs_folds_n = best_res_n["ibs"].values

    max_common = min(np.max(np.array(t)) for t in times_folds)
    min_common = max(np.min(np.array(t)) for t in times_folds)
    time_grid = np.linspace(min_common, max_common, 300)

    ibs_mean = np.mean(ibs_folds_n)
    ibs_std = np.std(ibs_folds_n)
    ibs_min = np.min(ibs_folds_n)
    ibs_max = np.max(ibs_folds_n)
    ibs_p25 = np.percentile(ibs_folds_n, 25)
    ibs_p50 = np.percentile(ibs_folds_n, 50)
    ibs_p75 = np.percentile(ibs_folds_n, 75)

    for i, (times, bs) in enumerate(zip(times_folds, brier_scores_n)):
        f = interp1d(times, bs, kind='nearest', bounds_error=False, fill_value=np.nan)
        ax2.plot(time_grid, f(time_grid), alpha=0.6, label=f"Fold {i}")

    ax2.set_title("Time-Dependent Brier Score across folds")
    ax2.set_xlabel("Time (days)")
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
    # MODEL n-LAYER → SHAP values
    # -----------------------------------------------------------
    explanation_pca_to_gene(ax_explain, models_n, X_pca_gpu, pca, gene_cols, clinical_cols,
                            results_dir + f"/shap_gene_{n}layers.csv")
    plt.tight_layout(pad=1.0)
    plt.show()
    fig.savefig(results_dir + f"/results_{n}layers.png")
    plt.close()
    print(f"Saved plot: {results_dir}/results_{n}layers.png \n")


def main():
    print("=" * 100, "Starting DeepSurv pipeline", "=" * 100, sep="\n")

    datasets = [
        #'miRNA/clinical_miRNA_normalized_log.csv',
        #'miRNA/clinical_miRNA_normalized_quant.csv',
        #'mRNA/clinical_mRNA_normalized_log.csv',
        'mRNA/clinical_mRNA_normalized_tpm_log.csv'
    ]

    network_selected = 3

    for dataset_file in datasets:
        # 1. Load data
        print("Preparing data...".center(100, '-'))
        dataset_name = os.path.basename(dataset_file).replace(".csv", "")
        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_file))
        subtype = dataset_file.split('/')[0]

        print(f"\nProcessing dataset: {dataset_name}. Subtype: {subtype}")

        # 2. Create necessary directories
        os.makedirs(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name), exist_ok=True)
        os.makedirs(os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name, 'models'), exist_ok=True)

        # 3. Prepare X, y and scale data
        X, y = prepare_data(dataset)
        X = scale_data(X, GENE_STARTS_WITH)

        gene_cols = [c for c in X.columns if c.startswith('hsa') or c.startswith('gene.')]
        clinical_cols = [c for c in X.columns if c not in gene_cols]

        # 4. STRATIFIED K-FOLD
        duration_bins = pd.qcut(y['duration'], q=4, labels=False)
        stratify_col = y['event'].astype(str) + "_" + duration_bins.astype(str)

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        #fold_indexes = list(kfold.split(X, y['event']))
        fold_indexes = list(kfold.split(X, stratify_col))

        # 5. PCA
        X_pca, pca, _ = fit_transform_pca(X, n_components=N_COMPONENTS)
        X_pca_gpu, y_gpu = data_to_gpu(X_pca, y, DEVICE)

        X_train_pca_folds = [X_pca_gpu[train_idx] for train_idx, _ in fold_indexes]
        X_val_pca_folds   = [X_pca_gpu[val_idx]   for _, val_idx   in fold_indexes]

        if network_selected == 3:
            # 6. Network 3 LAYERS
            param_grid_3 = {
                'hidden1': [64, 128, 256],
                'hidden2': [16, 32, 64],
                'dropout': [0.3, 0.5],
                'epochs': [500],
                'lr': [1e-2, 1e-3, 5e-4],
                'batch_size': [32, 64],
                'decay_lr': [0.003, 0.005],
                'weight_decay': [1e-4, 1e-5]
            }

            # 7. GRID SEARCH + CROSS-VALIDATION
            print("\nGrid search for best params...")
            gcv_best_3, _ = grid_searches(X_train_pca_folds, X_val_pca_folds, y_gpu, fold_indexes, subtype, Net_3layers,
                                          param_grid_3, dataset_name, ROOT, DEVICE)

            print("\nCross validation on best params...")
            models_3, cv_results_3 = cross_validate(X_train_pca_folds, X_val_pca_folds, y_gpu, fold_indexes,
                                                    gcv_best_3['best_params'], Net_3layers, subtype, dataset_name, ROOT,
                                                    DEVICE)

            # 8. PLOTS
            results_dir = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
            times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
            pca_plot(pca, gene_cols, results_dir)
            plots(models_3, cv_results_3, pca, X_pca_gpu, times_csv_path, gene_cols, clinical_cols, results_dir, n=3)

        else:
            # 6. Network 5 LAYERS
            param_grid_5 = {
                'hidden1': [128, 256], 'hidden2': [64, 128], 'hidden3': [32, 64], 'hidden4': [8, 16, 32],
                'dropout': [0.3, 0.5], 'lr': [0.05, 0.01, 0.001, 5e-4], 'batch_size': [32, 16],
                'epochs': [500], 'decay_lr': [0.003, 0.005], 'weight_decay': [1e-4, 1e-3, 1e-5]
            }

            # 7. GRID SEARCH + CROSS-VALIDATION
            print("\nGrid search for best params...")
            gcv_best_5, _ = grid_searches(X_train_pca_folds, X_val_pca_folds, y_gpu, fold_indexes, subtype, Net_5layers,
                                          param_grid_5, dataset_name, ROOT, DEVICE)

            print("\nCross validation on best params...")
            models_5, cv_results_5 = cross_validate(X_train_pca_folds, X_val_pca_folds, y_gpu, fold_indexes,
                                                    gcv_best_5['best_params'], Net_5layers, subtype, dataset_name, ROOT,
                                                    DEVICE)
            # 8. PLOTS
            results_dir = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
            times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
            plots(models_5, cv_results_5, pca, X_pca_gpu, times_csv_path, gene_cols, clinical_cols, results_dir, n=5)

        print("=" * 100, "Pipeline completed successfully.", "=" * 100, sep="\n")


if __name__ == "__main__":
    main()
