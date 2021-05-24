import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from get_ec import get_ec_1_level


def get_counts(column):
    counts = Counter(x for xs in column for x in set(xs))
    counts.most_common()
    # df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    # df_sorted = df.sort_values(by=[0], ascending=False)
    return counts.most_common()

def sequence_len(data):
    data['seq_len'] = data['sequence'].apply(lambda x: len(x))
    # data['seq_len'] = data['sequence'].str.len()

def plot_seq_len_distribution(df, data_name):
    sns.displot(df['seq_len'].values)
    plt.title(f'Seq len: {data_name}')
    plt.grid(True)
    return plt

def plot_seq_len_box_plot(data, ec_column):
    pass

if __name__ == '__main__':
    # read file
    hot_90 = pd.read_csv('/home/amsequeira/enzymeClassification/datasets/ecpred_uniprot_uniref_90.csv',
                         low_memory=False)
    data = hot_90.dropna(subset=['sequence'])
    print(hot_90.shape)
    # (175266, 8)

    # considering first 8 classes
    lev_1_multilabel = get_ec_1_level(hot_90, single_label=False)
    print(lev_1_multilabel)
    # [175266 rows x 9 columns] ec_number1

    # how many EC numbers different from 1 level
    multilabel = lev_1_multilabel.loc[lev_1_multilabel['ec_number1'].apply(len) > 2, :]
    # uniref_90  ... ec_number1
    # 84171     Q5I0K3  ...  [3, 4, 2]
    # 87044     Q8R4N0  ...  [3, 4, 2]
    # 87313     Q8N0X4  ...  [3, 4, 2]
    # 89108     Q9DBX6  ...  [4, 5, 1]
    # 89694     Q96SQ9  ...  [4, 5, 1]
    # 111873    Q28035  ...  [2, 5, 1]
    # 111876    P00502  ...  [2, 5, 1]
    # 113157    P08263  ...  [2, 5, 1]
    # [8 rows x 9 columns]

    # uniref_90   Entry  ... ec_number1 ec_single_label
    # 391       O28019  O28019  ...     [3, 4]          {3, 4}
    # 392       Q6HLE5  Q6HLE5  ...     [3, 4]          {3, 4}
    # 393       Q81G04  Q81G04  ...     [3, 4]          {3, 4}
    # 394       A7GMU8  A7GMU8  ...     [3, 4]          {3, 4}
    # 395       Q9K6Z4  Q9K6Z4  ...     [3, 4]          {3, 4}
    # ...     ...  ...        ...             ...
    # 150533    Q6WNV7  Q6WNV7  ...     [3, 2]          {3, 2}
    # 150691    P17255  P17255  ...     [3, 7]          {3, 7}
    # 151614    P53208  P53208  ...     [2, 3]          {2, 3}
    # 171153    P25373  P25373  ...     [2, 1]          {2, 1}
    # 171292    P17695  P17695  ...     [2, 1]          {2, 1}
    # [510 rows x 10 columns]

    counts_multilabel = get_counts(column=lev_1_multilabel['ec_number1'])
    # [('2', 54506),
    #  ('3', 34838),
    #  ('0', 22708),
    #  ('1', 19200),
    #  ('6', 16307),
    #  ('4', 13163),
    #  ('5', 8105),
    #  ('7', 6957)]

    lev_1_single_label = get_ec_1_level(hot_90, single_label=True)
    counts_single_label = get_counts(column=lev_1_single_label['ec_number1'])


    # [('2', 54343),
    #  ('3', 34484),
    #  ('0', 22708),
    #  ('1', 19066),
    #  ('6', 16290),
    #  ('4', 12924),
    #  ('5', 8081),
    #  ('7', 6860)]


    # taking single label into account check distribution of aa for class
    lev_1_single_label['ec_number1'] = [x[0] for x in lev_1_single_label['ec_number1']]
    labels = lev_1_single_label.ec_number1.unique()

    sequence_len(lev_1_single_label) # add seq_len column
    # table


    # plot seq len distribution plot
    # sns.displot(lev_1_single_label['seq_len'].values)
    # plt.title(f'Seq len: overall')
    # plt.grid(True)
    # plt.show()
    #
    # lev_1_single_label.groupby("ec_number1").seq_len.plot( kind='kde', legend=True)
    # plt.show()
    # plot seq len histogram plot
    # general

    # lev_1_single_label.seq_len.hist(alpha=0.4, range=[lev_1_single_label.seq_len.min(), lev_1_single_label.seq_len.max()])
    # plt.show()
    lev_1_single_label.seq_len.hist(alpha=0.4, range=[lev_1_single_label.seq_len.min(), 2000])
    plt.title('histogram of overall distribution of sequence len')
    plt.xlabel('Sequence length')
    plt.ylabel('Number of sequences')
    plt.savefig('/home/amsequeira/enzymeClassification/graphics/ec_90_single_overall_seq_len.png')
    plt.show()

    # per class
    lev_1_single_label.groupby("ec_number1").seq_len.hist(alpha=0.4, range=[lev_1_single_label.seq_len.min(), 1600],
                                                          histtype='step', stacked=True,fill=False,linewidth=2,legend=True)
    plt.title('histogram of distribution of sequence len per class')
    plt.xlabel('Sequence length')
    plt.ylabel('Number of sequences')
    plt.savefig('/home/amsequeira/enzymeClassification/graphics/ec_90_single_per_class_seq_len.png')
    plt.show()


    # plot seq len boxlot
    lev_1_single_label.boxplot(column = ['seq_len'])
    plt.title('boxplot of overall distribution of sequence len')
    plt.ylabel('sequence len')
    plt.savefig('/home/amsequeira/enzymeClassification/graphics/ec_90_single_overall_seq_len_box.png')
    plt.show()

    # per class
    lev_1_single_label.boxplot(column = ['seq_len'], by='ec_number1')
    plt.title('boxplot of overall distribution of sequence len')
    plt.ylabel('sequence len')
    plt.xlabel('ec class')
    plt.savefig('/home/amsequeira/enzymeClassification/graphics/ec_90_single_per_class_seq_len_box.png')
    plt.show()
    # plot all seq len all dataset boxplot
    # plot seq len per class boxplot
    lev_1_single_label.boxplot(column = ['seq_len'], by='ec_number1')
    plt.ylim(lev_1_single_label.seq_len.min(), 1600)
    plt.title('boxplot of overall distribution of sequence len')
    plt.ylabel('sequence len')
    plt.xlabel('ec class')
    plt.savefig('/home/amsequeira/enzymeClassification/graphics/ec_90_single_per_class_seq_len_box_reduce.png')
    plt.show()




    # fazer os mesmos graficos para o aas diferentes

    # data = get_ec_2_level_more_than_x_samples(hot_90, x=50, single_label=True)
    # data = get_ec_complete_more_than_x_samples(phys_90, x=30) # column to consider is 'ec_number4
