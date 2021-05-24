import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
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

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from nlf_blosum_encoding import blosum_encode
from sklearn.metrics import accuracy_score

# input is 21 categorical 200 paded aa sequences

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

# ## Z scales
zscale = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
    'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
}

# # BLOSUM
blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
}

hotencoded = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'C': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'G': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}


# get embedding weights

emb_20 = '/home/amsequeira/enzymeClassification/models/visualization/emb_500_post_20_ec90emb_wei'
emb_8 = '/home/amsequeira/enzymeClassification/models/visualization/emb_500_post_8_ec90emb_wei'
emb_5 = '/home/amsequeira/enzymeClassification/models/visualization/emb_500_post_5_ec90emb_wei'

alphabet="XARNDCEQGHILKMFPSTWYV"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
row_names = char_to_int.keys()

e20 = pd.read_csv(emb_20, index_col=0)
e8 = pd.read_csv(emb_8)
# e8['alphabet'] = row_names
# e8.set_index('alphabet')
e5 = pd.read_csv(emb_5)



# {0: 'X',
#  1: 'A',
#  2: 'R',
#  3: 'N',
#  4: 'D',
#  5: 'C',
#  6: 'E',
#  7: 'Q',
#  8: 'G',
#  9: 'H',
#  10: 'I',
#  11: 'L',
#  12: 'K',
#  13: 'M',
#  14: 'F',
#  15: 'P',
#  16: 'S',
#  17: 'T',
#  18: 'W',
#  19: 'Y',
#  20: 'V'}

b = pd.DataFrame.from_dict(blosum62, orient='index')
z = pd.DataFrame.from_dict(zscale, orient='index')
h = pd.DataFrame.from_dict(hotencoded, orient='index')

# cosine similarity
bcs=sklearn.metrics.pairwise.cosine_similarity(b)
bcs=pd.DataFrame(bcs, index=blosum62.keys(), columns=blosum62.keys())

zcs=sklearn.metrics.pairwise.cosine_similarity(z)
zcs=pd.DataFrame(zcs, index=zscale.keys(), columns=zscale.keys())

hcs=sklearn.metrics.pairwise.cosine_similarity(h)
hcs=pd.DataFrame(hcs, index=hotencoded.keys(), columns=hotencoded.keys())

# todo nao sei se deveria tirar o X do embedding. pôr e20[1:] e o alphabet sem o X
e20cs = sklearn.metrics.pairwise.cosine_similarity(e20)
e20cs=pd.DataFrame(e20cs, index=row_names, columns=row_names)

e8cs = sklearn.metrics.pairwise.cosine_similarity(e8)
e8cs=pd.DataFrame(e8cs, index=row_names, columns=row_names)

e5cs = sklearn.metrics.pairwise.cosine_similarity(e5)
e5cs=pd.DataFrame(e5cs, index=row_names, columns=row_names)

# "rocket", "mako", "flare", and "crest"
# "magma" and "viridis
# the annotate protein universe just compare cosine similarity with blosum62

bu = sns.heatmap(bcs, xticklabels=blosum62.keys(), yticklabels=blosum62.keys())
plt.title('cosine similarity blosum62')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_blosum62')
plt.show()

zu = sns.heatmap(zcs, xticklabels=zscale.keys(), yticklabels=zscale.keys())
plt.title('cosine similarity z scales')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_zscale')
plt.show()

hu = sns.heatmap(hcs, xticklabels=hotencoded.keys(), yticklabels=hotencoded.keys())
plt.title('cosine similarity hot encoded')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_ohe')
plt.show()


e20u = sns.heatmap(e20cs, xticklabels=char_to_int.keys(), yticklabels=char_to_int.keys())
plt.title('cosine similarity embedding 20 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb20')
plt.show()

e8u = sns.heatmap(e8cs, xticklabels=char_to_int.keys(), yticklabels=char_to_int.keys())
plt.title('cosine similarity embedding 8 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb8')
plt.show()

e5u = sns.heatmap(e5cs, xticklabels=char_to_int.keys(), yticklabels=char_to_int.keys())
plt.title('cosine similarity embedding 5 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb5')
plt.show()



# without cosine similarity (annotate universe compare with this one?)
bu = sns.heatmap(b, xticklabels=blosum62.keys(), yticklabels=blosum62.keys())
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/blosum62')
plt.title('blosum62')
plt.show()

# cosine similarity but put clustermap
buu = sns.clustermap(bcs,tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity blosum62')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusterblosum62')
plt.show()

zuu = sns.clustermap(zcs, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity z scales')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusterzscales')
plt.show()

huu = sns.clustermap(hcs, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity hot encoded')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusterohe')
plt.show()

e20uu = sns.clustermap(e20cs, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 20')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb20')
plt.show()

e8uu = sns.clustermap(e8cs, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 8')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb8')
plt.show()

e5uu = sns.clustermap(e5cs, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 5')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb5')
plt.show()

# without cosine similarity (annotate universe compare with this one?)
buu = sns.clustermap(b, tree_kws=dict(linewidths=1.5))
# plt.title('blosum62')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/clusterblosum')
plt.show()

# embedding without X
alphabet_nox="XARNDCEQGHILKMFPSTWYV"
char_to_int_nox = dict((c, i) for i, c in enumerate(alphabet_nox))
row_names_nox = char_to_int_nox.keys()

e20cs_nox = sklearn.metrics.pairwise.cosine_similarity(e20[1:])
e20cs_nox=pd.DataFrame(e20cs_nox, index=row_names_nox, columns=row_names_nox)

e8cs_nox = sklearn.metrics.pairwise.cosine_similarity(e8[1:])
e8cs_nox=pd.DataFrame(e8cs_nox, index=row_names_nox, columns=row_names_nox)

e5cs_nox = sklearn.metrics.pairwise.cosine_similarity(e5[1:])
e5cs_nox=pd.DataFrame(e5cs_nox, index=row_names_nox, columns=row_names_nox)


e20u_nox = sns.heatmap(e20cs_nox, xticklabels=char_to_int_nox.keys(), yticklabels=char_to_int_nox.keys())
plt.title('cosine similarity embedding 20 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb20_nox')
plt.show()

e8u_nox = sns.heatmap(e8cs_nox, xticklabels=char_to_int_nox.keys(), yticklabels=char_to_int_nox.keys())
plt.title('cosine similarity embedding 8 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb8_nox')
plt.show()

e5u_nox = sns.heatmap(e5cs_nox, xticklabels=char_to_int_nox.keys(), yticklabels=char_to_int_nox.keys())
plt.title('cosine similarity embedding 5 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_emb5_nox')
plt.show()

e20uu_nox = sns.clustermap(e20cs_nox, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 20')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb20_nox')
plt.show()

e8uu_nox = sns.clustermap(e8cs_nox, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 8')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb8_nox')
plt.show()

e5uu_nox = sns.clustermap(e5cs_nox, tree_kws=dict(linewidths=1.5))
# plt.title('cosine similarity embedding 5')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/cosine_clusteremb5_nox')
plt.show()


# grafico
def pca_scatter_plot(pca = bcs, title=' PCA Scatter plot'):
    index = pca.iloc[:,0].index
    for i in range(len(pca)):
        x = pca.iloc[i][0]
        y = pca.iloc[i][1]
        plt.plot(pca.iloc[:, 0], pca.iloc[:, 1], 'go', markersize=5)
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , index[i], fontsize=10)
        plt.title(title)
    return plt

# PCA
pca = sklearn.decomposition.PCA(n_components=2)


bcs=pca.fit_transform(b)
bcs=pd.DataFrame(bcs, index=blosum62.keys(), columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = bcs, title=' PCA Scatter plot blosum')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_blosum62')
plt.show()

zcs=pca.fit_transform(z)
zcs=pd.DataFrame(zcs, index=zscale.keys(), columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = zcs, title=' PCA Scatter plot Z scales')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_zscales')
plt.show()


hcs=pca.fit_transform(h)
hcs=pd.DataFrame(hcs, index=hotencoded.keys(), columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = hcs, title=' PCA Scatter plot hot encoded')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_ohe')
plt.show()

e20cs=pca.fit_transform(e20)
e20cs=pd.DataFrame(e20cs, index=row_names, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e20cs, title=' PCA Scatter plot embedding 20 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb20')
plt.show()


e8cs=pca.fit_transform(e8)
e8cs=pd.DataFrame(e8cs, index=row_names, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e8cs, title=' PCA Scatter plot embedding 8 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb8')
plt.show()


e5cs=pca.fit_transform(e5)
e5cs=pd.DataFrame(e5cs, index=row_names, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e5cs, title=' PCA Scatter plot embedding 5 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb5')
plt.show()



e20cs=pca.fit_transform(e20[1:])
e20cs=pd.DataFrame(e20cs, index=row_names_nox, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e20cs, title=' PCA Scatter plot embedding 20 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb20_nox')
plt.show()


e8cs=pca.fit_transform(e8[1:])
e8cs=pd.DataFrame(e8cs, index=row_names_nox, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e8cs, title=' PCA Scatter plot embedding 8 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb8_nox')
plt.show()


e5cs=pca.fit_transform(e5[1:])
e5cs=pd.DataFrame(e5cs, index=row_names_nox, columns=['PCA1', 'PCA2'])
pca_scatter_plot(pca = e5cs, title=' PCA Scatter plot embedding 5 dim')
plt.savefig('/home/amsequeira/enzymeClassification/graphics/heatmap_aa_encodings/pca_emb5_nox')
plt.show()


# euclidean distance (benchmarking paper plus heatmao)
# Does not give good results. anyway no article uses it
# # # tsne
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
#
# bcs=tsne.fit_transform(b)
# bcs=pd.DataFrame(bcs, index=blosum62.keys(), columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = bcs, title=' tsne Scatter plot blosum')
#
# zcs=tsne.fit_transform(z)
# zcs=pd.DataFrame(zcs, index=zscale.keys(), columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = zcs, title=' tsne Scatter plot Z scales')
#
# hcs=tsne.fit_transform(h)
# hcs=pd.DataFrame(hcs, index=hotencoded.keys(), columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = hcs, title=' tsne Scatter plot hot encoded')
#
# e20cs=tsne.fit_transform(e20)
# e20cs=pd.DataFrame(e20cs, index=row_names, columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = e20cs, title=' tsne Scatter plot embedding 20 dim')
#
# e8cs=tsne.fit_transform(e8)
# e8cs=pd.DataFrame(e8cs, index=row_names, columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = e8cs, title=' tsne Scatter plot embedding 8 dim')
#
# e5cs=tsne.fit_transform(e5)
# e5cs=pd.DataFrame(e5cs, index=row_names, columns=['tsne1', 'tsne2'])
# pca_scatter_plot(pca = e5cs, title=' tsne Scatter plot embedding 8 dim')
#
