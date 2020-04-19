import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def downsample_data(data):
    
    train_data, test_data = train_test_split(data,test_size = 0.2,random_state = 42, shuffle = True)
    low = train_data[train_data.PriorityID == 'Low']
    normal = train_data[train_data.PriorityID == 'Normal']
    high = train_data[train_data.PriorityID == 'High']
    critical = train_data[train_data.PriorityID == 'Critical']
    n = len(low)

    normal = normal.sample(n)
    high = high.sample(n)

    balanced_train_data = low.append(normal).append(high).append(critical)

    balanced_train_data = balanced_train_data.sample(frac=1).reset_index(drop=True)
    balanced_train_data['record_type'] = 'train'
    test_data['record_type'] = 'test'

    full_dataset = balanced_train_data.append(test_data)

    return full_dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data', 
                        type = str, 
                        help = 'Path to data to undersample.')

    args = parser.parse_args()

    data = pd.read_csv(args.data)
    dataset = downsample_data(data)
    
    dataset.to_csv(args.data[:-4]+'_downsample.csv')
