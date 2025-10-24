import pandas as pd
import numpy as np

import os
os.environ['RPY2_CFFI_MODE'] = 'ABI'

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

edgeR = importr('edgeR')

def tmm_normalization(counts_df):
    # Converte il DataFrame pandas in un oggetto R
    counts_r = pandas2ri.py2rpy(counts_df)
    
    # Crea un DGEList (oggetto edgeR)
    dge_list = edgeR.DGEList(counts=counts_r)
    
    # Calcola i fattori di normalizzazione TMM
    dge_list = edgeR.calcNormFactors(dge_list, method="TMM")
    
    # Applica la normalizzazione
    normalized_counts = edgeR.cpm(dge_list, normalized_lib_sizes=True)
    
    # Converte il risultato in un DataFrame pandas
    normalized_df = pandas2ri.rpy2py(normalized_counts)
    normalized_df.index = counts_df.index
    normalized_df.columns = counts_df.columns
    
    return normalized_df