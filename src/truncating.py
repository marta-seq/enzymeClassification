import pandas as pd
import numpy as np

def get_terminals(seq, l):
    if len(seq)<=l:
        return seq
    else:
        le=int(l/2)
        term_c=seq[:le]
        term_n=seq[-le:]
        terminals=term_c+term_n
        return terminals

def get_middle(seq,l):
    if int(len(seq))<=l:
        return seq
    else:
        takeoff = int(len(seq)-l)
        if (len(seq) % 2) == 0:
            takeoff_2 = takeoff//2
            term_c = takeoff_2
            term_n = takeoff_2
        else:
            takeoff_2 = takeoff//2
            term_c = takeoff_2
            term_n = takeoff_2 + 1
        middle = seq[term_c:-term_n]
        return middle

# check that are no sequences with more than l
# seq_new_list = []
# for seq in list_of_columns():
#     seq_new = get_middle(seq,len)
#     seq_new_list.append(seq_new)
# pd.DataFrame(seq_new_list).map(len).max()
# pad sequences




# df = df[['sequence'].str.split(',').map(len) < 500]