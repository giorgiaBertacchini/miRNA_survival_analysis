import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
import warnings
from sklearn.exceptions import FitFailedWarning
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import KernelPCA

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

base = os.path.basename(os.getcwd())
list = os.getcwd().split(os.sep) 
# list.pop(list.index(base))
ROOT = '\\'.join(list)
print(ROOT)
DATA_PATH = os.path.join(ROOT, 'datasets\\preprocessed')
SEED = 42
DATASET_NAME = 'miRNA\\clinical_miRNA_normalized_log.csv'
DATASET_TYPE = DATASET_NAME.split('\\')[-1][9:-4]
NUM_FOLDS = 10
USE_CLINICAL = True
DATASET_TYPE = 'clinical_'+DATASET_TYPE if USE_CLINICAL else 'seq_only'+DATASET_TYPE
SUBTYPE=DATASET_NAME.split("\\")[0]

def plot_coefficients(coefs):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    ax.legend(loc='best')

def c_index_scorer(estimator, X, y):
    # Estrai i campi 'death' e 'days' da y
    event_indicator = y['event']
    event_time = y['time']
    # Predici i punteggi di rischio
    risk_scores = estimator.predict(X)
    # Calcola il concordance index
    c_index = concordance_index_censored(event_indicator, event_time, risk_scores)[0]
    return c_index

def parse_array(x):
    if isinstance(x, str):
        x = x.strip('[]')
        return eval(x, {'np':np})

def prepare_data(dataset, use_clinical):
    y_cols = ['Death', 'days_to_death', 'days_to_last_followup']
    if use_clinical:
        print('USING clinical data') 
        X_cols = [col for col in dataset.columns if col not in y_cols]
    else:
        print('EXCLUDING clinical data') 
        miRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'hsa' not in col]
        mRNA_clinical_cols = [col for col in dataset.columns if col not in y_cols and 'gene.' not in col]
        X_cols = [col for col in dataset.columns if col not in y_cols and not col in miRNA_clinical_cols and col not in mRNA_clinical_cols]

    custom_dtype = np.dtype([
        ('event', np.bool_),      # O 'bool'
        ('time', np.float64)      # O 'float'
    ])

    y = []
    for index,row in dataset[y_cols].iterrows():
        if row['Death'] == 1:
            y.append(np.array((True, row['days_to_death'].item()), dtype=custom_dtype))
        elif row['Death'] == 0:
            tuple = (False, row['days_to_last_followup'].item())
            y.append(np.array(tuple, dtype=custom_dtype)) 
    y = np.array(y)

    X = dataset[X_cols]
    return X, y    

def scale_data(X):
    scaler = StandardScaler()

    genes_cols = [col for col in X.columns if 'hsa' in col or 'gene.' in col]
    genes_cols.append('age_at_initial_pathologic_diagnosis')
    scaled_X = pd.DataFrame(scaler.fit_transform(X[genes_cols]), columns=genes_cols)

    X[genes_cols] = scaled_X
    return X

def search_best_params(kfold, X_train, y_train):
    dir = os.path.join(ROOT, f"grid_searches\\cox_regression\\{SUBTYPE}\\{DATASET_TYPE}")
    os.makedirs(dir, exist_ok=True)

    bp_path = os.path.join(dir,"best_params.json")
    alphas_path = os.path.join(dir, "alphas.csv")

    cen = CoxnetSurvivalAnalysis()
    best_params = {}

    # search if already computed
    if os.path.exists(bp_path) and os.path.exists(alphas_path):
        print('Found best parameters')
        with open(bp_path, 'r') as f:
            best_params = json.load(f)
            res = pd.read_csv(alphas_path)
            res['param_alphas'] = res['param_alphas'].apply(parse_array)
            print(best_params)
        
    # Compute best parameters with grid search and random search    
    else:
        print("Computing new params from scratch")

        X_train_gen = X_train[[col for col in X_train.columns if col.startswith('hsa') or col.startswith('gene.')]]
        best_params['alpha_min_ratio'], best_params['l1_ratio'] = gcv(cen, kfold, X_train_gen, y_train)
        print(f"Computed best alpha_min_ratio {best_params['alpha_min_ratio']} and l1_ratio {best_params['l1_ratio']}")
        
        best_params['rcv'] = {}
        best_params['filtered'] = {}
        print("Computing best alphas")
        best_params['rcv']['best_alpha'], best_params['rcv']['cols'], res = rcv(kfold, X_train_gen, y_train, best_params['alpha_min_ratio'], best_params['l1_ratio'])
        
        print(f"Found possible best allpha with RCV: {best_params['rcv']['best_alpha']}")
        best_params['filtered']['best_alpha'], best_params['filtered']['cols'] = filter_alphas(res, best_params, X_train_gen, y_train)
        
        print(f"Found possible best alpha with our filtering: {best_params['filtered']['best_alpha']}")

        # saving best params
        print(f"Saving best_params into: {bp_path}")
        with open(bp_path, 'w') as f:
            json.dump(best_params, f)
        res.to_csv(alphas_path, index=False)

    return best_params, res

def gcv(cen, kfold, X_train, y_train):
    gcv = GridSearchCV(
        cen,
        param_grid={
            "l1_ratio": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "alpha_min_ratio":[0.01, 0.05, 0.1]
        },
        cv=kfold,
        error_score=0.5,
        n_jobs=8,
    ).fit(X_train, y_train)
    return gcv.best_params_['alpha_min_ratio'], gcv.best_params_['l1_ratio']

def rcv(kfold, X_train, y_train, alpha_min_ratio, l1_ratio, save_res=False):
    cen = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio)

    # warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)

    cen.fit(X_train, y_train)
    estimated_alphas = cen.alphas_
    rcv = RandomizedSearchCV(
        cen,
        param_distributions={
            "alphas":[[v] for v in map(float, estimated_alphas)]
        },
        cv=kfold,
        scoring=c_index_scorer,
        error_score=0.5,
        n_iter=30, # 30
        n_jobs=8,
        verbose=0,
        random_state=SEED
    ).fit(X_train, y_train)
    
    res = pd.DataFrame(rcv.cv_results_)
    if save_res:
        res.to_csv(os.path.join(ROOT, f"grid_searches\\{SUBTYPE}\\{DATASET_TYPE}_alphas.csv"), index=False)

    cols = X_train.columns[(rcv.best_estimator_.coef_ != 0)[:, 0]]
    return rcv.best_params_['alphas'], cols.values.tolist(), res
    
def filter_alphas(res, best_params, X_train, y_train):
    #filtering best alpha
    elegible_models = []
    min_cols_model = None
    for i in range(len(res['param_alphas'])):
        print(f'Testing alpha {i}/{len(res['param_alphas'])}')
        cen = CoxnetSurvivalAnalysis(l1_ratio = best_params['l1_ratio'], alpha_min_ratio = best_params['alpha_min_ratio'], alphas=[res['param_alphas'][i]], max_iter=100000)
        
        try:
            cen.fit(X_train, y_train)
        except Exception as e:
            if isinstance(e, ArithmeticError):
                print("ArithmeticError occurred")
                continue
            else:
                print(e)
                continue

        non0_coefs = cen.coef_[cen.coef_ != 0]
        cols = X_train.columns[(cen.coef_ != 0)[:, 0]]
        if min_cols_model is None or (len(non0_coefs) > 100 and len(non0_coefs) < min_cols_model['num_coefs']):
            min_cols_model={
                'model':cen,
                'num_coefs':len(non0_coefs),
                'filtered_best_alpha':res['param_alphas'][i],
                'score':res['mean_test_score'][i]
            }

        if len(non0_coefs) > 100 and len(non0_coefs) <=200:
            elegible_models.append({
                'model':cen,
                'num_coefs':len(non0_coefs),
                'filtered_best_alpha':res['param_alphas'][i],
                'score':res['mean_test_score'][i]
            })
        
        if len(elegible_models) != 0:
            best_model = elegible_models[0]
            for model in elegible_models:
                if model != best_model and model['score'] > best_model['score']:
                    best_model = model
        else:
            best_model = min_cols_model

    return best_model['filtered_best_alpha'], cols.values.tolist()

def run_test(best_params, res, kfold, X_train, y_train, X_test, y_test):
    to_test = ['rcv', 'filtered']
    dir = os.path.join(ROOT, f'results\\{SUBTYPE}\\{DATASET_TYPE}')
    os.makedirs(dir , exist_ok=True)
    cv_results_path = os.path.join(dir, "cv_results.txt.")
    
    clin_cols = [col for col in X_train.columns if 'hsa' not in col and 'gene.' not in col]
    
    if os.path.exists(cv_results_path):
        with open(cv_results_path, 'w') as f:
            f.write('')

    for source in to_test:
        msg = ''
        alpha = best_params[source]['best_alpha']

        # reduce again with PCA
        num_comps = 20
        if len(best_params[source]['cols']) > num_comps:
            kpca = KernelPCA(n_components=num_comps, kernel='rbf')
            train_data = pd.concat([X_train[clin_cols].reset_index(drop=True), pd.DataFrame(columns=[str(i) for i in range(1, num_comps+1)], data=kpca.fit_transform(X_train[best_params[source]['cols']]))], axis=1)
            test_data = pd.concat([X_test[clin_cols].reset_index(drop=True), pd.DataFrame(columns=[str(i) for i in range(1, num_comps+1)], data=kpca.transform(X_test[best_params[source]['cols']]))], axis=1)
        else:
            train_data = X_train[clin_cols + best_params[source]['cols']]
            test_data = X_test[clin_cols + best_params[source]['cols']]

        # fit and train cox for survival analysis
        print("One shot validation on test set with best parameters and reduced features")
        cen_best = CoxnetSurvivalAnalysis(alpha_min_ratio=best_params['alpha_min_ratio'], l1_ratio=best_params['l1_ratio'], alphas=alpha, max_iter=100000)#  #for predicting just the risk score
        cen_best.fit(train_data, y_train)
        msg += f"Finished training cen_best:\n\t{cen_best}\n"
        preds = cen_best.predict(test_data)
        msg += f"Predictions: {preds}\n"
        msg += f"Concordance index: {cen_best.score(test_data, y_test)}\n"

        msg+="Beginning cross validation\n"
        X = np.concatenate((train_data, test_data), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        cv = cross_val_score(
            CoxnetSurvivalAnalysis(alpha_min_ratio=best_params['alpha_min_ratio'], l1_ratio=best_params['l1_ratio'], alphas=alpha, max_iter=100000), 
            X, y, cv=kfold, scoring=c_index_scorer)
        msg += f"Cross validation scores: {cv}\n"
        msg += f"Cross validation description:\n\tmean={cv.mean()} \n\tstd={cv.std()} \n\tmin={cv.min()} \n\tmax={cv.max()}\n\n"

        print(msg)

        with open(cv_results_path, 'a') as f:
            f.write(msg)

        print(f"Best coefficients considered: {cen_best.coef_[cen_best.coef_ != 0]}\n {train_data.columns[np.where(cen_best.coef_ != 0)[0]]}")

        plots_to_generate = [
            (boxplot_train_results, ([res['mean_test_score']])),
            (plot_alphas, (res, best_params, source)),
            (plot_coefs, (cen_best.coef_[cen_best.coef_ != 0], train_data.columns[np.where(cen_best.coef_ != 0)[0]])),
        ]
        create_subplots(plots_to_generate, os.path.join(dir, f'{source}_test_plots.png'))

def create_subplots(plot_list, filename):
    # Create the figure and a 2x2 grid of Axes objects
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten the 2D array of axes objects into a 1D array for easier iteration
    axs = axs.flatten()

    if len(plot_list) > len(axs):
        print(f"Warning: Only plotting the first {len(axs)} plots as the grid size is 2x2.")

    # Iterate through the functions and axes, calling the function for the corresponding subplot
    for i, (plot_func, data) in enumerate(plot_list[:len(axs)]):
        # The asterisk (*) unpacks the data tuple before passing it to the plotting function
        plot_func(axs[i], *data)

    # Adjust layout to prevent subplot titles and labels from overlapping
    plt.tight_layout(pad=3.0)

    fig.savefig(filename)

def plot_alphas(ax, res, best_params, test):
    alphas = res['param_alphas']#.apply(lambda x: x[0])
    mean = res.mean_test_score
    std = res.std_test_score
    
    df = pd.DataFrame(data={'alphas':alphas, 'mean':mean, 'std':std})
    df = df.sort_values(by='alphas')

    alphas = df['alphas']
    mean = df['mean']
    std=df['std']

    ax.plot(alphas, mean)
    ax.set_title('Concorande-index value for each alpha tested')
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(best_params[test]['best_alpha'][0], c='C4')
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)

def boxplot_train_results(ax, data):
    ax.boxplot(data)
    ax.set_title('Boxplot of c-index values for each alpha tested')
    ax.set_ylabel('C-index')

def plot_coefs(ax, values, names):
    df = pd.DataFrame({"coefs":names, "values":values})
    df = df.sort_values(by='values', key=abs, ascending=True)

    df.plot.barh(ax=ax, legend=True)
    ax.set_title('Non 0 coefficients')
    ax.set_xlabel('coefficient value')
    ax.set_ylabel('coefficient index')
    ax.grid(True)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

        print('Searching best params for genes reads'.center(100, '-'))
        best_params, res = search_best_params(kfold, X_train, y_train)
        print("Finished computing best params, running test...".center(100, '-'))
        run_test(best_params, res, kfold, X_train, y_train, X_test, y_test)
        print("Finished cox regression script".center(100, '-'))

if __name__ == "__main__":
    main()