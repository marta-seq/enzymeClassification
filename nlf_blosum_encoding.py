import sys
import os
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append('/home/amsequeira/deepbio')

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '5,7'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# todo add these functions to propythia
########################################################################################################################
# NLF ENCODING
# This method of encoding is detailed by Nanni and Lumini in their paper.
# It takes many physicochemical properties and transforms them using a Fisher Transform (similar to a PCA)
# creating a smaller set of features that can describe the amino acid just as well. There are 19 transformed features.
#
# L. Nanni and A. Lumini, “A new encoding technique for peptide classification,”
# Expert Syst. Appl., vol. 38, no. 4, pp. 3185–3191, 2011
# https://dmnfarrell.github.io/bioinformatics/mhclearning
# The main difference between proteinand peptides is that the proteins have variable lengths while in agiven problem
# he peptides have a fixed length. For this reasonin the proteins it is very important to develop a system that considers
# also the sequence (as the pseudo amino acids encoding ofShenand Chou (2007)while in the peptides the sequence can be
# intrin-sically obtained by simply concatenating the features that de-scribes each amino acid
# each aminoacid has 18 phys values
# just receives 20 aa letters
nlf = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)
def nlf_encode(seq):
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)
    # show_matrix(x)
    e = x.values.flatten().tolist()
    return e

########################################################################################################################
# BLOSUM
# BLOSUM62 is a substitution matrix that specifies the similarity of one amino acid to another by means of a score.
# This score reflects the frequency of substiutions found from studying protein sequence conservation in large databases
# of related proteins. The number 62 refers to the percentage identity at which sequences are clustered in the analysis.
# Encoding a peptide this way means we provide the column from the blosum matrix corresponding to the amino acid at each
# position of the sequence. This produces 21x9 matrix.
# see https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/blosum

# blosum_62 =  pd.read_csv('/home/amsequeira/deepbio/src/mlmodels/blosum62.csv')
blosum_62 =  pd.read_csv('/home/amsequeira/enzymeClassification/blosum62_20.csv')

def blosum_encode(seq):
    #encode a peptide into blosum features
    x = pd.DataFrame([blosum_62[i] for i in seq]).reset_index(drop=True)
    # show_matrix(x)
    e = x.values.flatten().tolist()
    return e
