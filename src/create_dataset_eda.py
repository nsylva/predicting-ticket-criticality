import pandas as pd
import numpy as np
import nltk
import sklearn

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import io
import os
import math
import zipfile
import time

from tqdm import tqdm
from pprint import pprint

from tqdm import tqdm
from collections import namedtuple

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import gc
gc.collect() # Force garbage collector to release unreferenced memory


################################################################################
### FUNCTIONS
################################################################################

def text_company(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"greenshades support", "clientswsupport")
    df[text_field] = df[text_field].str.replace(r"greenshades software support", "clientswsupport")
    df[text_field] = df[text_field].str.replace(r"greenshades partner support", "partnerswsupport")
    df[text_field] = df[text_field].str.replace(r"greenshades ticket", "ticket")
    df[text_field] = df[text_field].str.replace(r"greenshades online", "")
    df[text_field] = df[text_field].str.replace(r"www.greenshades.com", "")
    df[text_field] = df[text_field].str.replace(r"greenshadesonline.com", "")
    df[text_field] = df[text_field].str.replace(r"greenshades", "")
    df[text_field] = df[text_field].str.replace(r"green shades", "")
    return df

def text_normalize(df, text_field):
    df[text_field] = df[text_field].str.replace(r"<[^>]+>", "") # Delete any string between "<" and ">"
    df[text_field] = df[text_field].str.replace(r"www.*?.com", "")
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"\d+", "") # Remove any strings of numbers
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9\.\=]", " ")
    return df

def tokens_to_string(input_list):
    return ' '.join(input_list)

def plot_cm(y_true, y_pred, figsize=(10,8)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    fig, ax = plt.subplots(figsize=figsize)
    colormap = sns.diverging_palette(220, 20, sep=20, as_cmap=True) #plt.cm.coolwarm #coolwarm_r reverse
    sns.heatmap(cm,
                cmap=colormap,
                annot=annot,
                annot_kws={"size": 12},
                fmt='',
                linewidths=1,
                square=True,
                ax=ax)
    ax.set_title('Confusion Matrix\n', fontsize=16, weight='bold');
    ax.set_xlabel('\n\nPredicted Label', fontsize=12);
    ax.set_ylabel('Actual Label\n\n', fontsize=12);
    plt.show()
    return

def plt_barchart(df, title):
    sns.set_style("white")
    df2 = df.groupby('PriorityID').agg({'PriorityID':['count']}).reset_index()
    df2.columns = ['PriorityID', 'count']
    df2['pct'] = df2['count']*100/(sum(df2['count']))
    x = df2['PriorityID']
    y = df2['pct']
    palette = ['red','orange', 'green', 'yellow']
    fig, ax = plt.subplots(figsize = (8,4))
    fig = sns.barplot(y, x, estimator = sum, ci = None, orient='h', palette=palette)

    for i, v in enumerate(y):
        ax.text(v+1, i+.05, str(round(v,3))+'%', color='black', fontweight='bold')

    ax.set(xlim=(0,100))
    plt.title(title + '\nTicket Priority as Percentage of Total', size=16, weight='bold')
    plt.ylabel('Ticket Priority')
    plt.xlabel('% Total');
    plt.show()
    return

def prt_counts(df, title):
    df2 = df.groupby('PriorityID').agg({'PriorityID':['count']}).reset_index()
    df2.columns = ['PriorityID', 'count']
    df2["pct"] = df2["count"]/(sum(df2["count"]))
    print(title)
    print("-"*30)
    print(df2)
    print("-"*30)
    total = df2["count"].sum()
    print("Total count = ", total)
    print()
    return

def plot_history(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    return

priority_dict = {'Low': 0,'Normal':1,'High':2,'Critical': 3}
def map_priority(priority, priority_dict = {}):
    val = priority_dict[priority]
    return val

def tokenize_note(note):
    '''Simple wrapper for tokenizing a note to its ids.'''
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(note))

def truncate_tokenized_note(tokenized_note,length):
    '''Truncates a tokenized note to the specified number of tokens(length).'''
    if len(tokenized_note) > length:
        tokenized_note = tokenized_note[:length]
    return tokenized_note

def create_tokenized_field(df, text_field, new_token_field):
    new_tokens = [[]]*len(df[text_field])
    for i in range(0, len(df[text_field])):
        tokens = word_tokenize(df[text_field][i])
        filtered_tokens = [w for w in tokens]
        new_tokens[i] = filtered_tokens
    df[new_token_field] = new_tokens
    return df

def releaseDF(df):
    '''
    Release dataframe from memory
    '''
    list_df = [df]
    del list_df
    return


def create_single_input(sentence, MAX_LEN):
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:MAX_LEN]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids, masks, segments

def create_input_array(sentences):
    input_ids, input_masks, input_segments = [], [], []

    for sentence in tqdm(sentences,position=0, leave=True):
        ids, masks, segments = create_single_input(sentence, MAX_SEQ_LEN-2)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]

def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


################################################################################
### GLOBAL VARIABLES
################################################################################
filename_in = 'baseline1_raw.csv'
filename_out = "baseline1_eda.csv"

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
    csv_filename = args.data

    ############################################################################
    ################################################################################
    # Ingest and Pre-Process Dataset
    ticket_data = pd.read_csv(csv_filename, dtype = {'Notes': str})
    ticket_data.Notes = ticket_data.Notes.fillna(' ')

    # count the number of empty notes
    count_empty = 0
    for i in range(0, len(ticket_data['Notes'])):
        note = ticket_data['Notes'][i].strip()
        if len(note) == 0:
            count_empty += 1

    # Ensure the dataframe is in the correct order. This should probably first be moved up to the data pre-processing.
    ticket_data = ticket_data.sort_values(['BugID','ActionOrder'])

    #group the data on the BugID and aggregate all of the features into lists for each feature. List maintains insertion order
    # following the sort on ActionOrder above.
    ticket_data_sequenced = ticket_data.groupby('BugID').aggregate(lambda attr: attr.tolist())

    # set up the Priorities at ints
    priority_dict = {'Low': 0,'Normal':1,'High':2,'Critical': 3}
    nan = ticket_data.PriorityID.unique()[4]
    priority_dict = {'Low': 0,'Normal':1,'High':2,'Critical': 3, 'normal': 1, nan: 1}

    ticket_data['PriorityID_Int'] = ticket_data.PriorityID.apply(map_priority, priority_dict = priority_dict)
    ticket_data["clean_notes"] = ticket_data.Notes
    ticket_data = text_company(ticket_data, "clean_notes")
    ticket_data = text_normalize(ticket_data, "clean_notes")
    ticket_data = create_tokenized_field(ticket_data, "clean_notes", "tokenized_notes")
    ticket_data["EmptyNote"] = ticket_data.tokenized_notes.apply(lambda x: 1 if len(x)==0 else 0)

    # Remove empty notes
    ticket_data = ticket_data[ticket_data.EmptyNote==0]
    ticket_data["note_length"] = ticket_data.tokenized_notes.apply(lambda x: len(x))

    # Prepare Dataset for BERT Layer
    cols = ['PriorityID', 'PriorityID_Int', 'note_length']
    dataset = ticket_data[cols]
    dataset = dataset.rename(columns={"PriorityID_Int": "label"})
    dataset.reset_index(inplace=True, drop=True)
    dataset = dataset.copy(deep=True)
    #releaseDF(ticket_data)
    #plt_barchart(dataset, "Full Dataset")

    # Save Pre-Processed Dataset

    # Save new dataframe to shared project data folder
    dataset.to_csv(args.data[:-8]+'_eda.csv')

    print()
    print("*"*80)
    print("FINISHED EDA DATASET")
    print("*"*80)
################################################################################
