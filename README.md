# FNP task1
Binary classification task

Text section contains a causal relation => 1, Otherwise => 0

Unbalanced dataset, only about 7% 1s

10837 training data & 2710 testing data

Submit your result on Kaggle

Read data descriptions and download the dataset from kaggle

## Code

1. `preprocess.py`: Separate the dataset into the training set, validation set, and test set.

2. Train the models via `baseline.py` with BERT & Focal Loss and generate the csv file `pred.csv`.

3. Generate pseudo-label training data via `create_pseudo.py` .

4. Train the models via `xlnet_pseudo.py` with XLNet, Focal Loss & pseudo-label.

5. Submit the result `final_preditcion`.
