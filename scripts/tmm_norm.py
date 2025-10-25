import os
os.environ['RPY2_CFFI_MODE'] = 'ABI'
os.environ['R_HOME'] = r"C:\\Program Files\\R\\R-4.4.2"
os.environ['R_USER'] = os.path.expanduser('~')

from rpy2.robjects import conversion, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
 
# Attiva la conversione tra pandas DataFrame e R DataFrame
conversion.set_conversion(pandas2ri.converter)

# Importa il pacchetto edgeR
edgeR = importr('edgeR')

def tmm_normalization(counts_df):
    # Usa il contesto di conversione locale per pandas e R
    with localconverter(pandas2ri.converter):
        # Converte il DataFrame pandas in un oggetto R
        counts_r = pandas2ri.py2rpy(counts_df)
    
    # Crea un DGEList (oggetto edgeR)
    dge_list = edgeR.DGEList(counts=counts_r)
    
    # Calcola i fattori di normalizzazione TMM
    dge_list = edgeR.calcNormFactors(dge_list, method="TMM")
    
    # Applica la normalizzazione
    normalized_counts = edgeR.cpm(dge_list, normalized_lib_sizes=True)
    
    # Converte il risultato in un DataFrame pandas
    with localconverter(pandas2ri.converter):
        normalized_df = pandas2ri.rpy2py(normalized_counts)
    
    # # Imposta gli indici e le colonne del DataFrame normalizzato
    normalized_df.index = counts_df.index
    normalized_df.columns = counts_df.columns
    
    return normalized_df

# Esempio di utilizzo
normalized_reads = tmm_normalization(genes_reads)
print(normalized_reads.head())