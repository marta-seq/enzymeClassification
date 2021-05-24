#
"""
get differents subsets of y
"""
import re
import pandas as pd
from collections import Counter
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder



def turn_single_label(column, data):
    l = []
    for ec_list in data[column]:
        ec_l = set(ec_list)
        l.append(ec_l)
    data['ec_single_label']=l
    data = data.loc[data['ec_single_label'].apply(len)<2,:]
    return data

def remove_zeros(column, data):
    list_zeros=[0, 0.0, '0', '0.0', '0.0.0', '0.0.0.0']
    l = []
    for ec_list in data[column]:
        ec_l = [x for x in ec_list if x not in (list_zeros)]
        l.append(ec_l)
    data['non_negative'] =l

    data = data.loc[data['non_negative'].apply(len)>0,:]
    return data



# BINARIZE LABELS
def binarize_labels(fps_y): # for single
    test = pd.Series(fps_y)

    # mlb = MultiLabelBinarizer()
    # hot = mlb.fit_transform(test)
    # res = pd.DataFrame(hot,
    #                    columns=mlb.classes_,
    #                    index=test.index)
    fps_y = [item for sublist in fps_y for item in sublist] # this line is because they are retrieved as a list
    encoder = LabelEncoder()
    encoder.fit(fps_y)
    encoded_Y = encoder.transform(fps_y)
    classes = encoder.classes_
    fps_y = np_utils.to_categorical(encoded_Y) # convert integers to dummy variables (i.e. one hot encoded)

    # print(fps_y)
    # print(fps_y.shape)

    from sklearn.preprocessing import OneHotEncoder
    # creating instance of one-hot-encoder
    # enc = OneHotEncoder(handle_unknown='ignore')
    # # passing bridge-types-cat column (label encoded values of bridge_types)
    # enc_df = pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
    # # merge with main df bridge_df on key values
    # bridge_df = bridge_df.join(enc_df)
    # bridge_df
    # bridge_types = list(set(fps_y))
    # bridge_df = pd.DataFrame(bridge_types, columns=['EC'])
    # # generate binary values using get_dummies
    # dum_df = pd.get_dummies(bridge_df, columns=["EC"], prefix=["Type_is"])
    # # merge with main df bridge_df on key values
    # bridge_df = bridge_df.join(dum_df)
    # print(bridge_df)

    return encoded_Y, fps_y, classes





# # select rows with complete ec numbers
# len(data['ec_number'].unique()) # 5315 different classes counts agglomerates
# data.groupby('ec_number').size().sort_values(ascending=False).head(20) # know which classe are most representative

# 1.Top 1000 classes complete classes with only 15. 800 with more than  MULTILABEL
def get_ec_complete_more_than_x_samples(data, x, single_label=True):
    # get only EC COMPLETE
    l = []
    for ec_list in data['ec_number']:
        ec_complete = [x.strip() for x in ec_list.split(';') if "-" not in x]
        l.append(list(set(ec_complete)))

    data['ec_number4'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number4'].apply(len)>0,:]  # (153672, 59)

    if single_label:
        data = turn_single_label('ec_number4', data)
    else:
        pass

    # result = {x for l in data['ec_number4'] for x in l} #4603 differennt unique values
    # result_2 = set(chain(*data['ec_number4'])) # 4603 unique values
    counts = Counter(x for xs in data['ec_number4'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)

    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 5
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] #20
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] #115
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 344
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 536
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 709
    # # + de 15
    # df_15 = df.loc[df[0]>15] #1069

    # + de x samples -

    df_15 = df.loc[df[0]>x] # 5

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number4']:
        ec = [x for x in ec_list if x in (list(df_15['index']))]
        l.append(ec)
    data['ec_number4'] = l # data 153672
    data = data.loc[data['ec_number4'].apply(len)>0,:] #142092
    return data   # column to consider is 'ec_number

##############
# get all the 3º level ec numbers complete with more than x samples
def get_ec_3_level_more_than_x_samples(data, x, single_label=True):
    # get all until the 3 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_3 = [re.match(r'.*(?=\.)',x).group(0) for x in ec_list.split(';') ]

        l.append(list(set(ec_3)))
    data['ec_number3']=l
    # get only 3º level complete (remove dashes)
    l = []
    for ec_list in data['ec_number3']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_number3'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number3'].apply(len)>0,:]  # 170191

    if single_label:
        data = turn_single_label('ec_number3', data)
    else:
        pass


    counts = Counter(x for xs in data['ec_number3'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #256 classes
    df_sorted = df.sort_values(by=[0], ascending=False)


    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 41
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] # 67
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] #95
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 131
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 160
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 95
    # # + de 15
    # df_15 = df.loc[df[0]>15] #199

    # - de 15
    # df_15 = df.loc[df[0]<15] 55


    # + de x samples -
    df_15 = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number3']:
        ec = [x for x in ec_list if x in(list(df_15['index']))]
        l.append(ec)
    data['ec_number3'] = l # data 153672
    data = data.loc[data['ec_number3'].apply(len)>0,:] #142092
    return data

##############
# get all the 2º level ec numbers complete with more than x samples
def get_ec_2_level_more_than_x_samples(data, x, single_label=True):
    # get all until the 2 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_2 = [re.search(r'[^.]*.[^.]*',x).group(0) for x in ec_list.split(';') ]
        # [^,]* = as many non-dot characters as possible,
        # . = a dot
        # [^.]* = as many non-dot characters as possible
        l.append(list(set(ec_2)))
    data['ec_number2']=l

    # get only 2º level complete (remove dashes)
    l = []
    for ec_list in data['ec_number2']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_number2'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number2'].apply(len)>0,:]  # 174521

    if single_label:
        data = turn_single_label('ec_number2', data)
    else:
        pass


    counts = Counter(x for xs in data['ec_number2'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #73 classes
    df_sorted = df.sort_values(by=[0], ascending=False)


    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 27
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] # 39
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] # 45
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 55
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 60
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 45
    # # + de 15
    # df_15 = df.loc[df[0]>15] # 64

    # - de 15
    # df_15 = df.loc[df[0]<15] 7


    # + de x samples -
    df_x = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number2']:
        ec = [x for x in ec_list if x in(list(df_x['index']))]
        l.append(ec)
    data['ec_to_keep'] = l # data 153672
    data = data.loc[data['ec_to_keep'].apply(len)>0,:]
    return data


##############
# get all the 1º level ec numbers complete
def get_ec_1_level(data, single_label=True):
    # get all until the 1 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        # [^,]* = as many non-dot characters as possible,
        # . = a dot
        l.append(list(set(ec_1)))
    data['ec_number1']=l
    if single_label:
        data = turn_single_label('ec_number1', data)
    else:
        pass

    counts = Counter(x for xs in data['ec_number1'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)

    # [('2', 54508),
    #  ('3', 34838),
    #  ('0', 23245),
    #  ('1', 19200),
    #  ('6', 16307),
    #  ('4', 13163),
    #  ('5', 8105),
    #  ('7', 6957)]

    return data

#############################3
# get all 4 levels independently 2º, 3º 4º
def get_n_ec_level(n, data, x, single_label=True):
    # get all n level
    n-=1
    l = []
    for ec_list in data['ec_number']:
        ec_l = [x.split('.')[n] for x in ec_list.split(';')]
        l.append(ec_l)
    data['ec_level']=l

    # get only level complete (remove dashes)
    l = []
    for ec_list in data['ec_level']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_level'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_level'].apply(len)>0,:]

    if single_label:
        data = turn_single_label('ec_level', data)
    else:
        pass

    #most common
    counts = Counter(x for xs in data['ec_level'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #73 classes
    df_sorted = df.sort_values(by=[0], ascending=False)

    # + de x samples -
    df_x = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_level']:
        ec = [x for x in ec_list if x in(list(df_x['index']))]
        l.append(ec)
    data['ec_to_keep'] = l # data 153672
    data = data.loc[data['ec_to_keep'].apply(len)>0,:] #142092
    return data

# PREDICTING THE SECOND KNOWING THE FIRST SHUFFLE THE REST OF DATASET FOR NEGATIVES
def get_second_knowing_first(data, first_level, single_label=True):
    # get rows with first level
    l = []
    for ec_list in data['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        l.append(list(set(ec_1)))
    data['ec_number1']=l

    # t exclude the rows without the first level desired
    l = []
    for ec_list in data['ec_number1']:
        ec_1 = [x for x in ec_list if x == str(first_level)]
        l.append(list(set(ec_1)))
    data['ec_number1']=l
    data_negative = data.loc[data['ec_number1'].apply(len)<1,:]
    data_positive = data.loc[data['ec_number1'].apply(len)>0,:]

    # extract the second digit
    l=[]
    for ec_list in data_positive['ec_number']:
        ec_l = [x.split('.')[1] for x in ec_list.split(';')]
        l.append(list(set(ec_l)))
    data_positive['ec_number2']=l

    # remove dashes
    l = []
    for ec_list in data_positive['ec_number2']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data_positive['ec_number2'] = l
    if single_label:
        data = turn_single_label('ec_number2', data)
    else:
        pass

    # for dataset negative,
    # decide how much negatives
    # to decide how much negatives
    counts = Counter(x for xs in data_positive['ec_number2'] for x in set(xs))
    x = counts.most_common()
    n = int(data_positive.shape[0]/len(x)) # the number negatives has the len of dataset divided by number of classes as if it was a even distribution, although this do not happen in the classes
    # n = x[3][1] # negatives has the third most class

    #create a dataset negative
    data_negative = data_negative.sample(n=n, random_state=5)
    data_negative['ec_number2'] = '0'
    #join the datasets
    final_data = pd.concat([data_positive,data_negative])
    return final_data # column to consider is 'ec_number2
