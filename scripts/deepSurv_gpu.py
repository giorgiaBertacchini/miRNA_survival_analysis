import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from functools import reduce
import operator

import torch
import torch.nn as nn
import torchtuples as tt
from torch.optim.lr_scheduler import StepLR

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid

from networks import Net_3layers, Net_5layers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

NUM_FOLDS = 5
SEED = 42
VERBOSE = True
USE_CLINICAL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
list = os.getcwd().split(os.sep)
list.pop()
# list.pop(list.index(base))
ROOT = '\\'.join(list)
DATA_PATH = os.path.join(ROOT, 'datasets\\preprocessed')

AVAILABLE_DATASETS = {
    "miRNA_log": os.path.join("miRNA", "clinical_miRNA_normalized_log.csv"),
    "miRNA_quant": os.path.join("miRNA", "clinical_miRNA_normalized_quant.csv"),
    "mRNA_log": os.path.join("mRNA", "clinical_mRNA_normalized_log.csv"),
    "mRNA_tpm_log": os.path.join("mRNA", "clinical_mRNA_normalized_tpm_log.csv")
}


##############
# Save model #
##############
def save_best_model(best_model, with_clinical, dataset, net_name):
    if with_clinical:
        filename = f'{dataset}__{net_name}__clinical.pth'
        path = os.path.join(ROOT, f'models/mlp/{filename}')
    else:
        filename = f'{dataset}__{net_name}__no_clinical.pth'
        path = os.path.join(ROOT, f'models/mlp/{filename}')
    
    torch.save(best_model.state_dict(), path)


################
# Prepare Data #
################
def prepare_data(dataset, use_clinical):
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

    genes_cols = [col for col in X.columns if 'hsa' in col or 'gene.' in col]
    genes_cols.append('age_at_initial_pathologic_diagnosis')
    scaled_X = pd.DataFrame(scaler.fit_transform(X[genes_cols]), columns=genes_cols)

    X[genes_cols] = scaled_X
    return X


def data_on_gpu(X_train, y_train, X_test, y_test):
    X_train_torch = torch.tensor(X_train.values, dtype=torch.float32, device=DEVICE)
    X_test_torch = torch.tensor(X_test.values, dtype=torch.float32, device=DEVICE)
    y_train_tuple = (
        torch.tensor(y_train['duration'].to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train['event'].to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    )
    y_test_tuple = (
        torch.tensor(y_test['duration'].to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test['event'].to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    )

    return X_train_torch, y_train_tuple, X_test_torch, y_test_tuple


def create_model(in_features, params, network_class):
    if network_class == Net_3layers:
        net = Net_3layers(in_features, 1, params['hidden1'], params['hidden2'], params['dropout'])
    else:
        net = Net_5layers(in_features, 1, params['hidden1'], params['hidden2'], params['hidden3'],
                          params['hidden4'], params['dropout'])
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(params['lr'])
    # model.optimizer.set_weight_decay(params['weight_decay'])
    return model


def surv(X, y, fold_indexes, network_class, dataset_name):
    in_features = X.shape[1]

    best_result = {
        'best_score':-1,
        'best_params':None,
        'best_estimator':None,
        'loss':None
    }

    if network_class == Net_3layers:
        param_grid = {
            'hidden1': [32, 64, 128, 256],
            'hidden2': [16, 32, 64, 128],
            'dropout': [0.1, 0.3, 0.5, 0.7],
            'epochs': [500],
            'lr': [1e-2, 1e-3],
            'batch_size': [32, 64]
        }
    else:
        param_grid = {
            'hidden1': [128, 256],
            'hidden2': [64, 128],
            'hidden3': [32, 64],
            'hidden4': [16, 32],
            'dropout': [0.3, 0.5],
            'lr': [0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'epochs': [200, 500],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'lr_factor': [0.7, 0.5]
        }

    results = []
    combs = reduce(operator.mul, (len(v) for v in param_grid.values()))
    print(f"Testing {combs} possible parameters combinations")
    i=1

    for params in ParameterGrid(param_grid):
        scores = []
        print(f"Testing {i}/{combs}".center(100, '='))
        i+=1
        for train_idx, val_idx in fold_indexes:
            ####################
            # Cross-Validation #
            ####################
            durations_train = y[0][train_idx]
            events_train = y[1][train_idx]
            y_train_fold = (durations_train, events_train)

            durations_val = y[0][val_idx]
            events_val = y[1][val_idx]
            y_val_fold = (durations_val, events_val)

            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]

            val = (X_val_fold, y_val_fold)

            ##################
            # Model Training #
            ##################
            model = create_model(in_features, params, network_class)

            callbacks = [tt.callbacks.EarlyStopping(patience=25)]

            #if events_val.sum() != 0:
            log = model.fit(
                X_train_fold,
                y_train_fold,
                batch_size=params['batch_size'], epochs=params['epochs'],
                callbacks=callbacks,
                verbose=False,
                val_data=(X_val_fold, y_val_fold)
            )

            ###############
            # Evaltuation #
            ###############
            _ = model.compute_baseline_hazards()
            surv_df = model.predict_surv_df(X_val_fold)
            ev = EvalSurv(surv_df, y_val_fold[0].cpu().numpy(), y_val_fold[1].cpu().numpy(), censor_surv='km')
            scores.append(ev.concordance_td())

        # results.append({'params': params, 'mean_concordance': np.mean(scores)})
        if np.mean(scores) > best_result['best_score']:
            best_result['best_score'] = np.mean(scores)
            best_result['best_params'] = params
            best_result['best_estimator'] = model
            best_result['loss'] = log
        
        results.append({
            'mean_concordance': np.mean(scores) if scores else None,
            'std_concordance': np.std(scores) if scores else None,
            'params': params,
        })
        results[-1] = results[-1] | {f"split{i}_test_score":scores[i] for i in range(len(scores))}

    results = pd.DataFrame(results)

    clean_name = dataset_name.replace("\\", "_").replace("/", "_")
    network_name = str(Net_3layers).split('.')[1].strip('\'>')
    out_dir = os.path.join(ROOT, "deepsurv_gcv_results", network_name)
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, f"{clean_name}.csv"), index=False)

    print("\n✅ Migliori parametri trovati:")
    print(f"Miglior concordanza: {best_result['score']}")
    print(f"Migliori parametri: {best_result['params']}")
    print(f"Concordanza media: {results['mean_test_score'].mean()}")
    return best_result


def main():
    datasets = [
        'miRNA\\clinical_miRNA_normalized_log.csv',
        'miRNA\\clinical_miRNA_normalized_quant.csv',
        'mRNA\\clinical_mRNA_normalized_log.csv',
        'mRNA\\clinical_mRNA_normalized_tpm_log.csv'
    ]

    for dataset_name in datasets:
        print("Preparing data...".center(100, '-'))
        dataset = pd.read_csv(os.path.join(DATA_PATH, dataset_name))
        X, y = prepare_data(dataset, USE_CLINICAL)
        X = scale_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                                            stratify=y['event'])

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_indexes = []

        for train_idx, val_idx in kfold.split(X_train, y_train['event']):
            fold_indexes.append((train_idx, val_idx))

        X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu = data_on_gpu(X_train, y_train, X_test, y_test)

        best = surv(X_train_gpu, y_train_gpu, fold_indexes, Net_3layers, dataset_name)
        save_best_model(best['best_estimator'], USE_CLINICAL, dataset_name, net_name="deepSurv_3")

        best = surv(X_train_gpu, y_train_gpu, fold_indexes, Net_5layers, dataset_name)
        save_best_model(best['best_estimator'], USE_CLINICAL, dataset_name, net_name="deepSurv_5")


if __name__ == "__main__":
    main()
