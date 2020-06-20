import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import argparse
import os

test_path = './data/test.csv'

def bertIn(path_):
    le = LabelEncoder()
    # read source data from csv file
    df_data = pd.read_csv(path_, sep=';', engine='python')
    df_test = pd.read_csv(test_path, sep=';', engine='python')

    df_test['Gold'] = 0
    # split into train, test
    #df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=0, stratify=df_data.Gold.values)

    # create a new dataframe for train, dev data
    df_bert = pd.DataFrame({'id': df_data['Index'],
                            'label': le.fit_transform(df_data['Gold']),
                            'alpha': ['a'] * df_data.shape[0],
                            'text': df_data['Text']})

    # split into train, dev
    df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.3, random_state=0)

    # create new dataframe for test data
    df_bert_test = pd.DataFrame({'id': df_test['Index'],
                                 'text': df_test['Text'],
                                 'label': df_test['Gold']})

    df_bert_train.to_csv('data/train_real.tsv', sep='\t', index=False, header=False)
    print("Train data loaded")
    df_bert_dev.to_csv('data/dev_real.tsv', sep='\t', index=False, header=False)
    print("Dev data loaded")
    df_bert_test.to_csv('data/test_real.tsv', sep='\t', index=False, header=False)
    print("Test data loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--infile', type=str, default='./data/fnp2020-fincausal-task1.csv', help='task 1 data input repo')
    parser.add_argument('--infile', type=str, default='./data/train.csv', help='task 1 data input repo')

    args = parser.parse_args()
    path_ = args.infile
    bertIn(path_)