import random
import os
from functools import reduce

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

from networks import Net_3layers, Net_5layers

# ---------------------- CONFIG ----------------------
NUM_FOLDS = 5
SEED = 42
USE_CLINICAL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_COMPONENTS = 50

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
DATA_PATH = os.path.join(ROOT, 'datasets\\preprocessed')


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
    in_features = N_COMPONENTS
    best_result = {'best_score': -1, 'best_params': None, 'best_model': None}

    total_combs = reduce(operator.mul, (len(v) for v in param_grid.values()))
    print(f"Testing {total_combs } possible parameters combinations")

    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        scores = []
        print(f"Testing {i}/{total_combs }".center(100, '='))
        for train_idx, val_idx in fold_indexes:
            X_train_fold_np = X[train_idx].cpu().numpy()
            X_val_fold_np = X[val_idx].cpu().numpy()

            X_train_pca, pca_model = fit_transform_pca(X_train_fold_np)
            X_val_pca = pca_model.transform(X_val_fold_np)

            X_train_fold = torch.tensor(X_train_pca, dtype=torch.float32, device=DEVICE)
            X_val_fold = torch.tensor(X_val_pca, dtype=torch.float32, device=DEVICE)

            y_train_fold = (y[0][train_idx], y[1][train_idx])
            y_val_fold = (y[0][val_idx], y[1][val_idx])
            #X_train_fold = X[train_idx]
            #X_val_fold = X[val_idx]

            model = create_model(in_features, params, network_class)
            #if events_val.sum() != 0:
            log = model.fit(X_train_fold, y_train_fold,
                            batch_size=params['batch_size'], epochs=params['epochs'],
                            callbacks=[tt.callbacks.EarlyStopping(patience=25)],
                            verbose=False, val_data=(X_val_fold, y_val_fold))

            _ = model.compute_baseline_hazards()
            surv_df = model.predict_surv_df(X_val_fold)
            ev = EvalSurv(surv_df, y_val_fold[0].cpu().numpy(), y_val_fold[1].cpu().numpy(), censor_surv='km')
            scores.append(ev.concordance_td())

        mean_score = np.mean(scores)
        results.append({'params': params, 'mean_concordance': mean_score, 'std_concordance': np.std(scores)})
        if mean_score > best_result['best_score']:
            best_result = {'best_score': mean_score, 'best_params': params, 'best_model': model, 'loss': log}

        results[-1] = results[-1] | {f"split{i}_test_score":scores[i] for i in range(len(scores))}

    results = pd.DataFrame(results)
    network_name = str(Net_3layers).split('.')[1].strip('\'>')
    results.to_csv(os.path.join(ROOT, f'\\deepsurv_gcv_results\\{network_name}\\{dataset_name}'), index=False)

    print("\n✅ Migliori parametri trovati:")
    print(f"Miglior concordanza: {best_result['score']}")
    print(f"Migliori parametri: {best_result['params']}")
    print(f"Concordanza media: {results['mean_concordance'].mean()}")
    return best_result, results


# ------------------ FINAL RETRAIN ------------------
def final_retrain(X_train_gpu, X_test_gpu, y_train_gpu, best_result, network_class):
    X_train_np = X_train_gpu.cpu().numpy()
    X_test_np = X_test_gpu.cpu().numpy()

    X_train_pca, pca_model = fit_transform_pca(X_train_np)
    X_test_pca = pca_model.transform(X_test_np)

    X_train_pca_gpu = torch.tensor(X_train_pca, dtype=torch.float32, device=DEVICE)
    X_test_pca_gpu = torch.tensor(X_test_pca, dtype=torch.float32, device=DEVICE)

    final_model = create_model(N_COMPONENTS, best_result['best_params'], network_class)
    final_model.fit(X_train_pca_gpu, y_train_gpu,
                    batch_size=best_result['params']['batch_size'],
                    epochs=best_result['params']['epochs'],
                    verbose=True)

    return final_model, pca_model, X_test_pca_gpu


# ------------------ SAVE MODEL ------------------
def save_model(final_model, pca_model, dataset_name, network_name, use_clinical=USE_CLINICAL):
    clinical_tag = "clinical" if use_clinical else "no_clinical"
    model_path = os.path.join(ROOT, 'models', 'mlp', f"{dataset_name}__{network_name}__{clinical_tag}.pth")
    torch.save(final_model.state_dict(), model_path)

    #pca_path = os.path.join(ROOT, 'models', 'mlp', f"{dataset_name}__{network_name}__{clinical_tag}_pca.pkl")
    #import joblib
    #joblib.dump(pca_model, pca_path)


def main():
    datasets = [
        'miRNA\\clinical_miRNA_normalized_log.csv',
        'miRNA\\clinical_miRNA_normalized_quant.csv',
        'mRNA\\clinical_mRNA_normalized_log.csv',
        'mRNA\\clinical_mRNA_normalized_tpm_log.csv'
    ]

    for dataset_file  in datasets:
        print("Preparing data...".center(100, '-'))
        dataset_name = os.path.basename(dataset_file).replace(".csv", "")
        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_name))
        X, y = prepare_data(dataset)
        X = scale_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                                            stratify=y['event'])
        X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu = data_to_gpu(X_train, y_train, X_test, y_test)

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = list(kfold.split(X_train, y_train['event']))

        # ---------------- 3 LAYERS ----------------
        param_grid_3 = {
            'hidden1': [32, 64, 128, 256], 'hidden2': [16, 32, 64, 128],
            'dropout': [0.1, 0.3, 0.5, 0.7], 'epochs': [500],
            'lr': [1e-2, 1e-3], 'batch_size': [32, 64]
        }
        best_3, results_3 = cross_validate(X_train_gpu, y_train_gpu, fold_indexes, Net_3layers, param_grid_3,
                                           dataset_name)
        final_model_3, pca_model_3, X_test_pca_3 = final_retrain(X_train_gpu, X_test_gpu, y_train_gpu, best_3,
                                                                 Net_3layers)
        save_model(final_model_3, pca_model_3, dataset_name, "deepSurv_3")

        # ---------------- 5 LAYERS ----------------
        param_grid_5 = {
            'hidden1': [128, 256], 'hidden2': [64, 128], 'hidden3': [32, 64], 'hidden4': [16, 32],
            'dropout': [0.3, 0.5], 'lr': [0.01, 0.001, 0.0001], 'batch_size': [32, 64, 128],
            'epochs': [200, 500], 'weight_decay': [1e-6, 1e-5, 1e-4], 'lr_factor': [0.7, 0.5]
        }
        best_5, results_5 = cross_validate(X_train_gpu, y_train_gpu, fold_indexes, Net_5layers, param_grid_5,
                                           dataset_name)
        final_model_5, pca_model_5, X_test_pca_5 = final_retrain(X_train_gpu, X_test_gpu, y_train_gpu, best_5,
                                                                 Net_5layers)
        save_model(final_model_5, pca_model_5, dataset_name, "deepSurv_5")


if __name__ == "__main__":
    main()
