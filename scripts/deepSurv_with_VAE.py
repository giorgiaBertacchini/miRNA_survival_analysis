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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import matplotlib.pyplot as plt

from networks import Net_3layers, Net_5layers

# Fix for scipy version compatibility
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

"""
DeepSurv pipeline with VAE-based dimensionality reduction,
cross-validated hyperparameter search, and SHAP-based model interpretability.
"""

# ---------------------- CONFIGURATION ----------------------
NUM_FOLDS = 10  # Number of CV folds TODO aumentare?
SEED = 42  # Random seed for reproducibility
USE_CLINICAL = True  # Whether to include clinical variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on device: {DEVICE}\n")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Paths
base = os.path.basename(os.getcwd())
list_path = os.getcwd().split(os.sep)
# list_path.pop()
# list.pop(list.index(base))
ROOT = '\\'.join(list_path)
# ROOT = os.path.dirname(os.getcwd()) + "/mbulgarelli"
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
        RNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'VAE_' not in col]
        X_cols = [col for col in dataset.columns if
                  col not in y_cols and not col in RNA_clinical_cols]

    X = dataset[X_cols].copy()
    return X, y


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


# ---------------------- MODEL ----------------------
def create_model(X, params, network_class):
    """
    Instantiates a DeepSurv model (CoxPH) with the specified
    neural network architecture and hyperparameters.
    """
    in_features = X.shape[1]
    if network_class == Net_3layers:
        net = Net_3layers(in_features, 1, params['hidden1'], params['hidden2'], params['dropout'])
    else:
        net = Net_5layers(in_features, 1, params['hidden1'], params['hidden2'], params['hidden3'],
                          params['hidden4'], params['dropout'])
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(params['lr'])

    return model


# ------------------ CALLBACKS ------------------
class LrLogger(tt.callbacks.Callback):
    def on_epoch_end(self):
        lr = self.model.optimizer.param_groups[0]['lr']
        print(f"LR = {lr:.6e}")


class DecayLR(tt.callbacks.Callback):
    def __init__(self, lr0, decay_rate=0.003):
        self.lr0 = lr0
        self.decay_rate = decay_rate
        self.epoch = 0

    def on_epoch_start(self):
        epoch = self.epoch
        new_lr = self.lr0 / (1.0 + epoch * self.decay_rate)

        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.epoch += 1


# ------------------ CROSS-VALIDATION ------------------
def grid_searches(X_train_folds, X_val_folds, y, fold_indexes, subtype, network_class, param_grid,
                  dataset_name):
    print("=" * 100)
    print(f"Starting grid search for {network_class.__name__}")
    print("=" * 100)

    best_result = {'best_score': -1, 'best_params': None, 'best_model': None}
    total_combs = reduce(operator.mul, (len(v) for v in param_grid.values()))
    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        cindex_scores = []
        print(f"Testing {i}/{total_combs}".center(100, '-'))
        print(f"Parameters: {params}")

        for (train_idx, val_idx), X_train, X_val in zip(fold_indexes, X_train_folds, X_val_folds):
            X_train_fold = X_train.to(DEVICE, dtype=torch.float32)
            X_val_fold = X_val.to(DEVICE, dtype=torch.float32)

            y_train_fold = (y[0][train_idx], y[1][train_idx])
            y_val_fold = (y[0][val_idx], y[1][val_idx])

            # model = create_model(in_features, params, network_class)
            model = create_model(X_train_fold, params, network_class)

            lr_decay_cb = DecayLR(lr0=params['lr'], decay_rate=params['decay_lr'])
            model.fit(X_train_fold, y_train_fold,
                      batch_size=params['batch_size'], epochs=params['epochs'],
                      callbacks=[
                          tt.callbacks.EarlyStopping(patience=25),
                          lr_decay_cb,
                          # LrLogger()
                      ],
                      verbose=False, val_data=(X_val_fold, y_val_fold))

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

    print("\nBest hyperparameters identified.")
    print(f"\tBest concordance index: {best_result['best_score']:.4f}")
    print(f"\tOptimal parameter set: {best_result['best_params']}")

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


def cross_validate(X_train_folds, X_val_folds, y, fold_indexes, params, network_class, subtype, dataset_name):
    print(f"\nRunning cross-validation for {network_class.__name__}\n")
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

    for fold_idx, ((train_idx, val_idx), X_train, X_val) in enumerate(
            zip(fold_indexes, X_train_folds, X_val_folds)):
        print(f"Training fold {fold_idx + 1}/{len(fold_indexes)}")

        X_train_fold = X_train.to(DEVICE, dtype=torch.float32)
        X_val_fold = X_val.to(DEVICE, dtype=torch.float32)

        y_train_fold = (y[0][train_idx], y[1][train_idx])
        y_val_fold = (y[0][val_idx], y[1][val_idx])

        # model = create_model(in_features, params, network_class)
        model = create_model(X_train_fold, params, network_class)

        lr_decay_cb = DecayLR(lr0=params['lr'], decay_rate=params['decay_lr'])
        log = model.fit(X_train_fold, y_train_fold,
                        batch_size=params['batch_size'], epochs=params['epochs'],
                        callbacks=[
                            tt.callbacks.EarlyStopping(patience=25),
                            lr_decay_cb
                        ],
                        verbose=False, val_data=(X_val_fold, y_val_fold))

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


def explanation_to_gene(ax, models, X_gpu, gene_cols, clinical_cols, file_path, top_k=30):
    print("Computing SHAP values...")

    X = X_gpu.detach().float().to(DEVICE)
    all_shap_projected = []

    for model in models:
        net = WrappedNet(model.net, DEVICE).to(DEVICE)
        net.eval()

        bg_idx = np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)
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
        shap_vae = shap_values[:, n_clinical:]

        shap_full = np.concatenate([shap_clinical, shap_vae], axis=1)
        all_shap_projected.append(shap_full)

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

    # Store all SHAP values and compute their mean across models
    all_shap_values = np.array(all_shap_values)  # shape: (n_models, n_samples, n_features)
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

    # Convert to numpy array
    mean_shap_across_models = np.tensordot(weights, all_shap_values, axes=(0, 0))  # shape: (n_samples, n_features)

    # Calc mean |SHAP| per feature
    feature_names = list(feature_names)
    mean_abs_shap = np.mean(np.abs(mean_shap_across_models), axis=0)

    # Order features by importance
    top_idx = np.argsort(mean_abs_shap)[::-1][:30]
    top_values = mean_abs_shap[top_idx]
    top_features = [str(feature_names[i]) for i in top_idx]

    # Plot
    ax.barh(top_features[::-1], top_values[::-1], color='skyblue')
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top Feature Importances (SHAP) - DeepSurv")
    ax.invert_yaxis()


# ------------------- PLOTS -------------------
def plots(models_n, best_res_n, X_gpu, times_csv_path, gene_cols, clinical_cols, results_dir, n=3):
    
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

    # -----------------------------------------------------------
    # MODEL n-LAYER
    # -----------------------------------------------------------
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
    explanation_to_gene(ax_explain, models_n, X_gpu, gene_cols, clinical_cols,
                            results_dir + f"/shap_gene_{n}layers.csv")
    plt.tight_layout(pad=1.0)
    plt.show()
    fig.savefig(results_dir + f"/results_{n}layers.png")
    plt.close()
    print(f"Saved plot: {results_dir}/results_{n}layers.png \n")


def main():
    print("=" * 100)
    print("Starting DeepSurv pipeline")
    print("=" * 100)

    datasets = [
        'miRNA/VAE_clinical_miRNA_normalized_log.csv',
        #'miRNA/VAE_clinical_miRNA_normalized_quant.csv',
        #'mRNA/VAE_clinical_mRNA_normalized_log.csv',
        #'mRNA/VAE_clinical_mRNA_normalized_tpm_log.csv'
    ]

    network_selected = 5

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

        gene_cols = [c for c in X.columns if c.startswith('hsa') or c.startswith('gene.')]
        clinical_cols = [c for c in X.columns if c not in gene_cols]

        # ---------------- STRATIFIED K-FOLD ----------------
        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = list(kfold.split(X, y['event']))

        # ---------------- ON GPU ----------------
        X_gpu, y_gpu = data_to_gpu(X, y)

        X_train_folds = []
        X_val_folds = []

        for train_idx, val_idx in fold_indexes:
            X_train_fold = X_gpu[train_idx]
            X_val_fold = X_gpu[val_idx]

            X_train_folds.append(X_train_fold)
            X_val_folds.append(X_val_fold)

        if network_selected == 3:
            # ---------------- 3 LAYERS ----------------
            """param_grid_3 = {
                'hidden1': [128, 256, 512],
                'hidden2': [32, 64, 128],
                'dropout': [0.3, 0.5],
                'epochs': [150, 300],
                'lr': [0.05, 1e-2, 1e-3],
                'batch_size': [32, 64, 128],
                'decay_lr': [0.003, 0.005]
            }"""
            param_grid_3 = {
                'hidden1': [256],
                'hidden2': [32],
                'dropout': [0.5],
                'epochs': [300],
                'lr': [0.01],
                'batch_size': [64],
                'decay_lr': [0.003]
            }
            print("\nGrid search for best params...")
            gcv_best_3, gcv_results_3 = grid_searches(X_train_folds, X_val_folds, y_gpu, fold_indexes, subtype,
                                                      Net_3layers, param_grid_3, dataset_name)

            print("\nCross validation on best params...")
            models_3, cv_results_3, df_times = cross_validate(X_train_folds, X_val_folds, y_gpu,
                                                              fold_indexes, gcv_best_3['best_params'],
                                                              Net_3layers, subtype, dataset_name)

            # ---------------- PLOTS ----------------
            results_dir = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
            os.makedirs(os.path.dirname(results_dir), exist_ok=True)
            times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
            plots(models_3, cv_results_3, X_gpu, times_csv_path, gene_cols,
                  clinical_cols, results_dir, n=3)

        else:
            # ---------------- 5 LAYERS ----------------
            """param_grid_5 = {
                'hidden1': [128, 256], 'hidden2': [64, 128], 'hidden3': [32, 64], 'hidden4': [8, 16, 32],
                'dropout': [0.3, 0.5], 'lr': [0.05, 0.01, 0.001], 'batch_size': [32, 128],
                'epochs': [300], 'decay_lr': [0.003, 0.005]
            }"""
            param_grid_5 = {
                'hidden1': [128], 'hidden2': [64, 128], 'hidden3': [32], 'hidden4': [16],
                'dropout': [0.3], 'lr': [0.01], 'batch_size': [64],
                'epochs': [150], 'decay_lr': [0.003]
            }

            print("\nGrid search for best params...")
            gcv_best_5, gcv_results_5 = grid_searches(X_train_folds, X_val_folds, y_gpu, fold_indexes, subtype,
                                                      Net_5layers, param_grid_5, dataset_name)

            print("\nCross validation on best params...")
            models_5, cv_results_5, df_times = cross_validate(X_train_folds, X_val_folds, y_gpu,
                                                              fold_indexes, gcv_best_5['best_params'],
                                                              Net_5layers, subtype, dataset_name)
            # ---------------- PLOTS ----------------
            results_dir = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name)
            os.makedirs(os.path.dirname(results_dir), exist_ok=True)
            times_csv_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv")
            plots(models_5, cv_results_5, X_gpu, times_csv_path, gene_cols,
                  clinical_cols, results_dir, n=5)

        print("=" * 100)
        print("Pipeline completed successfully.")
        print("=" * 100)


if __name__ == "__main__":
    main()
