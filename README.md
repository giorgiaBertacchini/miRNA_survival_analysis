# The miRNA Signature: Unveiling Breast Cancer Survival through different Cox Architectures

This project presents a comparative study on breast cancer (TCGA-BRCA) survival prediction using microRNA (miRNA) and mRNA expression profiles. The primary objective is to validate miRNA as a reliable prognostic biomarker compared to the mRNA-based state-of-the-art, utilizing advanced high-dimensional Cox regression techniques.

## 📖 Project Overview

Accurate survival prediction is essential for personalized oncology. While mRNA is the most common biomarker, this study explores how miRNA can offer granular risk stratification. We implemented and compared several models to handle high-dimensional omic data ($p \gg n$):

- Penalized linear models.

- Deep Learning architectures.

- Generative models (Variational Autoencoders).

## 📊 Dataset

Data was sourced from the Genomic Data Commons (GDC) focusing on the TCGA-BRCA cohort:

- miRNA-Seq: microRNA expression profiles.

- mRNA-Seq: Messenger RNA expression profiles (used as a benchmark).

- Clinical Data: Vital status, follow-up time, age, and tumor stage.

### Pre-processing

Various normalization and filtering techniques were applied:

- $log_2$ transformation and Quantile normalization.

- Feature selection via "Filtered Alpha" strategy.

- Dimensionality reduction through PCA, Kernel PCA, and Latent Space representation.

## 🛠 Methodological Pipeline

The project compares three main architectures:

1. ElasticNet + Kernel PCA: A penalized linear approach leveraging non-linear transformations to capture complex relationships between features.

2. DeepSurv (MLP): A deep neural network designed to model the Cox log-hazard, tested with 3-layer and 5-layer architectures.

3. CoxVAE + DeepSurv: A cutting-edge generative approach using a Variational Autoencoder with a survival-regularized loss function to learn more informative latent representations.

## 🏆 Key Results

- miRNA Performance: miRNA signatures demonstrated excellent prognostic capability, reaching a C-index of up to 0.882.

- Comparison with mRNA: While mRNA maintains higher absolute predictive power, miRNA offers superior calibration of survival probabilities (lower Integrated Brier Score) in specific Deep Learning scenarios.

- Non-linearity: The integration of non-linear components (Kernel PCA and VAE) significantly improved model robustness compared to traditional linear methods.

# Code execution
1. **(OPTIONAL)** Given the high dimension of the raw data, it's impossible for us to load it on github, but it can be downloaded from the GDC data portal (https://portal.gdc.cancer.gov/) by first building the cohort as described in the paper, and then by going into the repository and setting the filters again as displayed in the paper. After that follow the execution of the notebook [data_processing](notebooks/data_processing/data_processing.ipynb). The results of this should be 2 files [datasets\preprocessed\clinical_miRNA(RC_RPM).csv](datasets/preprocessed/clinical_miRNA(RC_RPM).csv) and [datasets\preprocessed\mRNA\clinical_mRNA(protein_coding).csv](datasets/preprocessed/mRNA/clinical_mRNA(protein_coding).csv). To see by yourself the analysis on data that we made on the paper, also those [notebooks](notebooks/data_analysis) are available.

2. **(OPTIONAL)** After that to obtain the normalized versions starting from these 2 files, follow the respective notebooks [normalization_miRNA](notebooks/normalization_miRNA.ipynb) and [normalization_mRNA](notebooks/normalization_mRNA.ipynb)

2. We directly provide the final pre-processed datasets that join clinical data with miRNA and mRNA genetic data, and, under the corresponding genetic source, and we provide both the simple normalized versions, and the normalized VAE-encoded versions.

3. Given the datasets, to reproduce the results, execute the 3 main scripts:

    - For [ElasticNet+KPCA](scripts/cox_regression.py) 
    ```bash 
    python cox_regression.py
    ```
    - For [PCA+DeepSurv](scripts/deepSurv.py)
    ```bash
    python deepSurv.py
    ```
    - For [VAE+DeepSurv](scripts/deepSurv_with_VAE.py)
    ```bash
    python deepSurv_with_VAE.py
    ```
4. Each one of these will use the best parameters computed by us in the grid searches we performed: in order to avoid it and recompute the parameters, delete or move folders inside [grid_searches](grid_searches)

5. The final results of the models will be inside the [results folder](results) for the penalized cox and KPCA approach, while inside the [deepsurv_results folder](deepsurv_results) for both the DeepSurv-based approaches.

NB:
- Plots are automatically created from the results and placed in the same folders
- In the ElasticNet plots the top left one represents the C-index scores obtained from the Random Search for the best alpha, and not from the final evaluation of the model with the best parameters. These latter results are not plotted and are only written inside the cv_results.txt files, and are the same that are in the tables of the paper appendix.