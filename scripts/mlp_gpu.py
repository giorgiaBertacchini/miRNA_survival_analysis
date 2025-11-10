import pandas as pd
import os
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler, EarlyStopping

from sksurv.metrics import concordance_index_censored

from networks import Net_3layers, Net_5layers

NUM_FOLDS = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
_ = torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

base = os.path.basename(os.getcwd())
path_parts = os.getcwd().split(os.sep)
path_parts.pop(path_parts.index(base))
ROOT = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(ROOT, 'datasets', 'preprocessed')

AVAILABLE_DATASETS = {
    "miRNA_log": "clinical_miRNA_normalized_log.csv",
    "miRNA_quant": "clinical_miRNA_normalized_quant.csv",
    "mRNA_log": os.path.join("mRNA", "clinical_mRNA_normalized_log.csv"),
    "mRNA_tpm_log": os.path.join("mRNA", "clinical_mRNA_normalized_tpm_log.csv")
}


####################
# Plot and results #
####################
def plot_real_vs_predicted(trues, preds, text):
    trues_flat = trues.flatten()
    preds_flat = preds.flatten()
    errors = preds_flat - trues_flat

    plt.figure(figsize=(7, 7))
    plt.scatter(trues_flat, preds_flat, c=np.abs(errors), cmap='viridis', alpha=0.7)
    plt.plot([trues.min(), trues.max()],
             [trues.min(), trues.max()],
             'r--', label='Perfect prediction')
    plt.colorbar(label="Absolute Error")
    plt.xlabel("Real Days to Death")
    plt.ylabel("Predicted Days to Death")
    plt.title(f"Predicted vs True Days to Death (colored by error) - {text}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_losses_grid(model, text):
    plt.figure(figsize=(10, 6))
    plt.plot(model.history[:, 'train_loss'], label='Train Loss')
    plt.plot(model.history[:, 'valid_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves - {text}')
    plt.legend()
    plt.grid()
    plt.show()


def print_results(preds, y_test_mlp):
    print("Metrics for log results:")

    mae = mean_absolute_error(y_test_mlp, preds)
    r2 = r2_score(y_test_mlp, preds)
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")

    print("Preds mean/std:", np.mean(preds), np.std(preds))


##############
# Save model #
##############
def save_model(model, with_clinical, folder, best_params, best_score, file):
    os.makedirs(f'../models/{folder}', exist_ok=True)

    if with_clinical:
        path = f'../models/{folder}/{file}_clinical.pkl'
    else:
        path = f'../models/{folder}/{file}_no_clinical.pkl'
    joblib.dump(model, path)

    net = model.module_
    params_dict = {}
    for name, param in net.named_parameters():
        params_dict[name] = param.detach().cpu().numpy()
    if with_clinical:
        np.savez(f'../models/{folder}/{file}_clinical.npz', **params_dict)
    else:
        np.savez(f'../models/{folder}/{file}_no_clinical.npz', **params_dict)

    # Write txt with best parameters
    txt_path = f'../models/mlp_path.txt'
    with open(txt_path, 'a') as f:
        f.write(f"Model path: {path}\n")
        f.write(f"Best parameters:\n")
        for key, value in best_params.items():
            f.write(f"\t{key}: {value}\n")
        f.write(f"Best score: {best_score}\n")
        f.write("=====================================\n")


################
# Prepare Data #
################
def prepare_data(dataset_path, with_clinical):
    dataset = pd.read_csv(dataset_path)

    y_cols = ['Death', 'days_to_death', 'days_to_last_followup']
    X_cols = [col for col in dataset.columns if col not in y_cols]

    if not with_clinical:
        # remove clinical data columns
        X_cols = [col for col in X_cols if col.startswith('hsa')]
    X = dataset[X_cols]

    custom_dtype = np.dtype([
        ('death', np.bool_),
        ('days', np.float64)
    ])

    y = []
    for index,row in dataset[y_cols].iterrows():
        if row['Death'] == 1:
            y.append(np.array((True, row['days_to_death'].item()), dtype=custom_dtype))
        elif row['Death'] == 0:
            y.append(np.array((False, row['days_to_last_followup'].item()), dtype=custom_dtype))
    y = np.array(y)

    death = y['death']
    days = y['days']
    y_signed = np.where(death, days, -days)

    #############
    # Z-scaling #
    #############
    cols_leave = [col for col in X.columns if col.startswith('pathologic')]
    cols_standardize = [col for col in X.columns if col not in cols_leave]

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    scaled_X = pd.DataFrame(
        x_mapper.fit_transform(X).astype('float32'),
        columns=X.columns,
        index=X.index
    )

    ##################
    # Data splitting #
    ##################
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y_signed, test_size=0.2, random_state=SEED)

    X_mlp = X_train.values.astype('float32')
    y_mlp = y_train.astype('float32')
    X_test_mlp = X_test.values.astype('float32')
    y_test_mlp = y_test.astype('float32')
    # Note: skotch converts data to tensor internally

    return X_mlp, y_mlp, X_test_mlp, y_test_mlp


def network(mlp_class, device, X_mlp):
    net = NeuralNetRegressor(
        module=mlp_class,
        module__input_dim=X_mlp.shape[1],
        module__output_dim=1,
        max_epochs=500,
        lr=1e-2,
        batch_size=32,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-6,
        criterion=nn.MSELoss(),
        device=device,
        iterator_train__drop_last=True,  # To ensure consistent batch sizes
        callbacks=[
            ('lr_scheduler', LRScheduler(
                policy=ReduceLROnPlateau,
                mode='min',
                factor=0.7,
                patience=10,
                monitor='valid_loss',
                min_lr=1e-6
            )),
            ('early_stopping', EarlyStopping(
                monitor='valid_loss',
                patience=25,
                threshold=1e-3,
                threshold_mode='rel',
                load_best=True
            ))
        ],
    )

    if mlp_class == Net_3layers:
        params = {
            'module__hidden1': [64, 128, 256],
            'module__hidden2': [32, 64, 128],
            'module__dropout': [0.3, 0.5],
            'optimizer__weight_decay': [1e-6, 1e-5, 1e-4],
            'callbacks__lr_scheduler__factor': [0.7, 0.5],
            'lr': [1e-2, 1e-3],
            'max_epochs': [250, 500],
            'batch_size': [16, 32, 64],
        }
    else:
        params = {
            'module__hidden1': [128, 256],
            'module__hidden2': [64, 128],
            'module__hidden3': [32, 64],
            'module__hidden4': [16, 32],
            'module__dropout': [0.3, 0.5],
            'max_epochs': [250, 500],
            'optimizer__weight_decay': [1e-6, 1e-5, 1e-4],
            'callbacks__lr_scheduler__factor': [0.7, 0.5],
            'lr': [1e-2, 1e-3],
            'batch_size': [16, 32, 64],
        }

    return net, params


def grid_search(mlp_class, X_mlp, y_mlp, device, kfold):
    """def c_index_scorer(y_true, y_pred):
        events = y_true > 0
        times = np.abs(y_true)
        return concordance_index_censored(events, times, y_pred)[0]

    cindex = make_scorer(c_index_scorer, greater_is_better=True)"""

    net, params = network(mlp_class, device, X_mlp)

    rs = GridSearchCV(
        estimator=net,
        param_grid=params,
        refit=True,
        cv=kfold,
        #scoring=cindex,  # If MSE Loss, comment this
        verbose=1,
        n_jobs=1
    )

    rs.fit(X_mlp, y_mlp)

    print("Best hyperparameters:", rs.best_params_)
    print("Best score:", rs.best_score_)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


def flow(data_type, with_clinical, device, kfold):
    dataset_path = os.path.join(DATA_PATH, AVAILABLE_DATASETS[data_type])
    X_mlp, y_mlp, X_test_mlp, y_test_mlp = prepare_data(dataset_path, with_clinical)

    print(f"[Net_3layers] Starting processing {data_type} ({'with' if with_clinical else 'without'} clinical data)")
    best_model, best_params, best_score = grid_search(Net_3layers, X_mlp, y_mlp, device, kfold)
    save_model(best_model, with_clinical, data_type, best_params, best_score, file="mlp_3")

    print(f"[Net_5layers] Starting processing {data_type} ({'with' if with_clinical else 'without'} clinical data)")
    best_model, best_params, best_score = grid_search(Net_5layers, X_mlp, y_mlp, device, kfold)
    save_model(best_model, with_clinical, data_type, best_params, best_score, file="mlp_5")


def main(data_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for with_clinical_data in [True, False]:
        flow(data_type, with_clinical_data, device, kfold)


if __name__ == "__main__":
    for data in ["miRNA_log", "miRNA_quant", "mRNA_log", "mRNA_tpm_log"]:
        main(data)
