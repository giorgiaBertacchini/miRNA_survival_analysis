import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torchtuples as tt
from torch.optim.lr_scheduler import StepLR

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from networks import Net_3layers, Net_5layers


base = os.path.basename(os.getcwd())
list = os.getcwd().split(os.sep)
list.pop(list.index(base))
ROOT = '\\'.join(list)
DATA_PATH = os.path.join(ROOT, 'datasets\\preprocessed')

AVAILABLE_DATASETS = {
    "miRNA_log": "clinical_miRNA_normalized_log.csv",
    "miRNA_quant": "clinical_miRNA_normalized_quant.csv",
    "mRNA_log": os.path.join("mRNA", "clinical_mRNA_normalized_log.csv"),
    "mRNA_tpm_log": os.path.join("mRNA", "clinical_mRNA_normalized_tpm_log.csv")
}

NUM_FOLDS = 5
SEED = 42
VERBOSE = True

random.seed(SEED)
np.random.seed(SEED)
_ = torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


##############
# Save model #
##############
def save_best_model(best_model, best_params, with_clinical, folder, file):
    if with_clinical:
        path = f'../models/{folder}/{file}_clinical.pkl'
    else:
        path = f'../models/{folder}/{file}_no_clinical.pkl'
    joblib.dump(best_model, path)

    net = best_model.net
    params_dict = {}
    for name, param in net.named_parameters():
        params_dict[name] = param.detach().cpu().numpy()
    if with_clinical:
        np.savez(f'../models/{folder}/{file}_clinical.npz', **params_dict)
    else:
        np.savez(f'../models/{folder}/{file}_no_clinical.npz', **params_dict)

    # Write txt with best parameters
    txt_path = f'../models/{folder}/deepSurv_clinical.txt'
    with open(txt_path, 'w') as f:
        f.write(f"Model on dataset: {folder}\n")
        f.write(f"Best parameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")


################
# Prepare Data #
################
def prepare_data(path, with_clinical, device):
    dataframe = pd.read_csv(path)
    dataframe = dataframe.rename(columns={'Death': 'event', 'days_to_last_followup': 'duration'})
    dataframe.drop(columns=['days_to_death'], inplace=True)

    if not with_clinical:
        # remove clinical data columns
        clinical_cols = [col for col in dataframe.columns if
                         not col.startswith('hsa') and col not in ['duration', 'event']]
        dataframe.drop(columns=clinical_cols, inplace=True)

    #############
    # Z-scaling #
    #############
    cols_leave = [col for col in dataframe.columns if col.startswith('pathologic')]
    cols_standardize = [col for col in dataframe.columns if col not in cols_leave + ['duration', 'event']]

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    y = (dataframe['duration'].values, dataframe['event'].values)

    scaled_X = x_mapper.fit_transform(dataframe).astype('float32')
    scaled_X = pd.DataFrame(scaled_X, columns=[col for col in dataframe.columns if col not in ['duration', 'event']])

    ##################
    # Data splitting #
    ##################
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=SEED)

    ###############
    # Data on GPU #
    ###############
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_train = (torch.tensor(y_train[0], dtype=torch.float32).to(device),
               torch.tensor(y_train[1], dtype=torch.float32).to(device))
    y_test = (torch.tensor(y_test[0], dtype=torch.float32).to(device),
              torch.tensor(y_test[1], dtype=torch.float32).to(device))

    return X_train, y_train, X_test, y_test


def surv(X, y, kfold, network_class):
    in_features = X.shape[1]

    if network_class == Net_3layers:
        param_grid = {
            'hidden1': [64, 128, 256],
            'hidden2': [32, 64, 128],
            'dropout': [0.3, 0.5],
            'lr': [0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'epochs': [200, 500],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'lr_factor': [0.7, 0.5]
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
    for params in ParameterGrid(param_grid):
        scores = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = (y[0][train_idx], y[1][train_idx])
            y_val = (y[0][val_idx], y[1][val_idx])

            if network_class == Net_3layers:
                net = Net_3layers(in_features, params['hidden1'], params['hidden2'], params['dropout'])
            else:
                net = Net_5layers(in_features, params['hidden1'], params['hidden2'], params['hidden3'],
                                  params['hidden4'], params['dropout'])

            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(params['lr'])
            model.optimizer.set_weight_decay(params['weight_decay'])
            scheduler = StepLR(model.optimizer, step_size=10, gamma=params['lr_factor'])
            callbacks = [tt.callbacks.EarlyStopping(patience=20), tt.callbacks.LRScheduler(scheduler)]

            _ = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                          callbacks=callbacks, verbose=False, val_data=(X_val, y_val))

            surv_df = model.predict_surv_df(X_val)
            ev = EvalSurv(surv_df, y_val[0].cpu().numpy(), y_val[1].cpu().numpy(), censor_surv='km')
            scores.append(ev.concordance_td())

        results.append({'params': params, 'mean_concordance': np.mean(scores)})

    best = max(results, key=lambda x: x['mean_concordance'])
    print("\n✅ Migliori parametri trovati:")
    print(best)
    return best


def flow(data_type, with_clinical, device, kfold):
    dataset_path = os.path.join(DATA_PATH, AVAILABLE_DATASETS[data_type])
    X_train, y_train, X_test, y_test = prepare_data(dataset_path, with_clinical, device)

    best = surv(X_train, y_train, kfold, Net_3layers)
    save_best_model(best['model'], best['params'], with_clinical, data_type, file="deepSurv_3")

    best = surv(X_train, y_train, kfold, Net_5layers)
    save_best_model(best['model'], best['params'], with_clinical, data_type, file="deepSurv_3")


def main(data_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for with_clinical_data in [True, False]:
        flow(data_type, with_clinical_data, device, kfold)


if __name__ == "__main__":
    for data in ["miRNA_log", "miRNA_quant", "mRNA_log", "mRNA_tpm_log"]:
        main(data)
