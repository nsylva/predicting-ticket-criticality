################################################################################
### Set up Environment
################################################################################
import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.model_selection import train_test_split

import gc
gc.collect() # Force garbage collector to release unreferenced memory

################################################################################
################################################################################
### MAIN
################################################################################
################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type = str,
                        help = 'Path to raw input dataset.')
    args = parser.parse_args()
    filename_in = args.data

    ############################################################################
    dataset = pd.read_csv(filename_in, dtype = {'Notes': str})
    dataset.isnull().any()
    dataset.reset_index
    dataset = dataset.rename(columns={'Unnamed: 0': 'OrigIndex'})
    dataset = dataset.dropna(how="any")
    dataset.reset_index(inplace=True, drop=True)

    ### Train/Test Split
    train, test = train_test_split(dataset,
                                   test_size=0.2,
                                   shuffle=True,
                                   random_state=273)

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    train_OrigIndex_list = list(train.OrigIndex)
    dataset['record_type'] = dataset['OrigIndex'].apply(lambda x: "train" if x in train_OrigIndex_list else "test")

    ### Random Downsampling to Equalize Low, Normal and High Priority Tickets
    train = dataset[dataset.record_type=="train"]
    test = dataset[dataset.record_type=="test"]

    df_low = train[train.PriorityID == 'Low']
    df_norm = train[train.PriorityID == 'Normal']
    df_high = train[train.PriorityID == 'High']
    df_crit = train[train.PriorityID == 'Critical']

    ## Resample training dataset based on "Low" PriorityID count
    # Downsample to equalize "Low", "Normal", and "High" PriorityID count
    n = len(df_low)
    df_low = df_low.sample(n)
    df_norm = df_norm.sample(n)
    df_high = df_high.sample(n)

    train = df_low.append(df_norm)
    train = train.append(df_high)
    train = train.append(df_crit)

    train = train.sample(frac=1, random_state = 273).reset_index(drop=True)

    df_new = train.append(test)
    df_new = df_new.sort_values(by='record_type', ascending=False)
    df_new.reset_index(inplace=True, drop=True)

    ############################################################################
    ### Save Balanced Dataset
    ############################################################################
    # Save new dataframe to shared project data folder
    dataset = df_new.copy()
    dataset.to_csv(args.data[:-17]+'_bal_downsampled.csv')

    print()
    print("*"*80)
    print("FINISHED CREATING DOWNSAMPLED DATASET")
    print("*"*80)
################################################################################
