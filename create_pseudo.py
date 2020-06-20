import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

test = pd.read_csv('data/test.csv', sep=';', dtype = 'str')
pred = pd.read_csv('pred.csv', dtype = 'str')
test['Gold'] = pred['Gold']
df_train, df_test = train_test_split(test, test_size=0.15, random_state=0, stratify=test.Gold.values)

train = pd.read_csv('data/train_real.tsv', sep='\t',header=None, names=['id', 'label', 'alpha', 'sentence'])


le = LabelEncoder()
df_bert = pd.DataFrame({'id': df_test['Index'],
                        'label': le.fit_transform(df_test['Gold']),
                        'alpha': ['a'] * df_test.shape[0],
                        'sentence': df_test['Text']})

pd.concat([train, df_bert]).reset_index(drop=True).to_csv('data/train_pseudo.tsv', sep='\t', index=False, header=False)

