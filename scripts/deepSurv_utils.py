import random
import os
import json
import torch
import numpy as np
import pandas as pd
from functools import reduce
import operator

import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Network custom architectures
from networks import Net_3layers, Net_5layers


def reproducibility(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def path_config():
    base = os.path.basename(os.getcwd())
    list_path = os.getcwd().split(os.sep)
    list_path.pop()
    ROOT = '\\'.join(list_path)
    DATA_PATH = os.path.join(ROOT, 'datasets/preprocessed')

    return ROOT, DATA_PATH


# ------------ DATA PREPARATION ------------
def prepare_data(dataset):
    """
    Splits the dataset into feature matrix X and survival targets y.
    Optionally includes or excludes clinical variables.
    """
    y_cols = ['Death', 'days_to_last_followup', 'days_to_death']
    y = dataset[['Death', 'days_to_last_followup']].copy()
    y = y.rename(columns={'Death': 'event', 'days_to_last_followup': 'duration'})

    X_cols = [col for col in dataset.columns if col not in y_cols]
    X = dataset[X_cols].copy()
    return X, y


def scale_data(X, gene_starts_with=("hsa", "gene.")):
    """
    Applies standard scaling to molecular features (miRNA/mRNA) and age at diagnosis.
    """
    scaler = StandardScaler()
    gene_cols = [c for c in X.columns if c.startswith(gene_starts_with)]
    gene_cols.append('age_at_initial_pathologic_diagnosis')
    scaled_X = pd.DataFrame(scaler.fit_transform(X[gene_cols]), columns=gene_cols)

    X[gene_cols] = scaled_X
    return X


def data_to_gpu(X, y, device):
    """
    Converts feature matrix and survival targets to PyTorch tensors
    and moves them to the selected device (CPU/GPU).
    """
    X_np = X.values if hasattr(X, "values") else X
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = (torch.tensor(y['duration'].to_numpy(dtype='float32'), device=device),
           torch.tensor(y['event'].to_numpy(dtype='float32'), device=device))
    return X_t, y_t


# --------------- MODEL CREATION ---------------
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

    model = CoxPH(
        net,
        tt.optim.Adam(
            lr=params['lr'],
            weight_decay=params.get('weight_decay', 1e-4)
        )
    )

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
def grid_searches(X_train_pca_folds, X_val_pca_folds, y, fold_indexes, subtype, network_class, param_grid,
                  dataset_name, ROOT, device):
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

        for (train_idx, val_idx), X_train_pca, X_val_pca in zip(fold_indexes, X_train_pca_folds, X_val_pca_folds):
            X_train_fold = X_train_pca.to(device, dtype=torch.float32)
            X_val_fold = X_val_pca.to(device, dtype=torch.float32)

            y_train_fold = (y[0][train_idx], y[1][train_idx])
            y_val_fold = (y[0][val_idx], y[1][val_idx])

            model = create_model(X_train_fold, params, network_class)

            lr_decay_cb = DecayLR(lr0=params['lr'], decay_rate=params['decay_lr'])
            model.fit(X_train_fold, y_train_fold,
                      batch_size=params['batch_size'], epochs=params['epochs'],
                      callbacks=[
                          tt.callbacks.EarlyStopping(patience=30, file_path=f"/tmp/deepsurv_{network_class.__name__}_{X_train_fold.shape[1]}_{random.random()}.pt"),
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

        torch.cuda.empty_cache()

    print("\nBest hyperparameters identified.")
    print(f"\tBest concordance index: {best_result['best_score']:.4f}")
    print(f"\tOptimal parameter set: {best_result['best_params']}")

    net_name = network_class.__name__
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, f'gcv_results_{net_name}.csv'),
                   index=False)

    best_path = os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name,
                             f'gcv_best_results_{net_name}.json')
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=4)

    return best_result, results


# ---------------- CROSS-VALIDATION ----------------
def cross_validate(X_train_pca_folds, X_val_pca_folds, y, fold_indexes, params, network_class, subtype, dataset_name,
                   ROOT, device):
    print(f"Running cross-validation for {network_class.__name__}\n")
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

    for fold_idx, ((train_idx, val_idx), X_train_pca, X_val_pca) in enumerate(
            zip(fold_indexes, X_train_pca_folds, X_val_pca_folds)):
        print(f"Training fold {fold_idx + 1}/{len(fold_indexes)}")

        X_train_fold = X_train_pca.to(device, dtype=torch.float32)
        X_val_fold = X_val_pca.to(device, dtype=torch.float32)

        y_train_fold = (y[0][train_idx], y[1][train_idx])
        y_val_fold = (y[0][val_idx], y[1][val_idx])

        model = create_model(X_train_fold, params, network_class)
        lr_decay_cb = DecayLR(lr0=params['lr'], decay_rate=params['decay_lr'])

        log = model.fit(X_train_fold, y_train_fold,
                        batch_size=params['batch_size'], epochs=params['epochs'],
                        callbacks=[
                            tt.callbacks.EarlyStopping(patience=30),
                            lr_decay_cb
                        ],
                        verbose=False, val_data=(X_val_fold, y_val_fold))

        _ = model.compute_baseline_hazards()
        surv_df = model.predict_surv_df(X_val_fold)

        durations_test = y_val_fold[0].cpu().numpy()
        ev = EvalSurv(surv_df,
                      durations_test,
                      y_val_fold[1].cpu().numpy(),
                      censor_surv='km')

        # C-index
        cindex_scores.append(ev.concordance_td())

        # Brier score
        t_max = np.percentile(durations_test, 95)
        time_grid = np.linspace(durations_test.min(), t_max, 100)
        bs = ev.brier_score(time_grid)
        brier_scores.append(list(bs))
        times_folds.append(list(time_grid))

        # Integrated Brier Score
        ibs = ev.integrated_brier_score(time_grid)
        ibs_scores.append(ibs)

        # Save model
        models.append(model)

        torch.cuda.empty_cache()

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
    fold_result.to_csv(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name,
                     f'cv_results_{network_class.__name__}.csv'),
        index=False)

    df_times.to_csv(os.path.join(ROOT, 'grid_searches', 'deepsurv', subtype, dataset_name, "times_by_fold.csv"),
                    index=False)

    # Save models
    for i, model in enumerate(models):
        save_model(model, subtype, dataset_name, network_name=network_class.__name__, ROOT=ROOT, index=i)

    return models, fold_result


# ------------------ SAVE MODEL ------------------
def save_model(final_model, subtype, dataset_name, network_name, ROOT, index=0):
    dict_path = os.path.join(ROOT, 'deepsurv_results', subtype, dataset_name, 'models')
    model_path = os.path.join(dict_path, f"{network_name}__fold{index}.pth")

    torch.save(final_model.net.state_dict(), model_path)
