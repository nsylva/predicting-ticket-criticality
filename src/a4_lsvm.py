################################################################################
### Set up Environment
################################################################################
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

# Suppress new window for plots
# %matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import gc
gc.collect() # Force garbage collector to release unreferenced memory

################################################################################
### Functions
################################################################################
def plot_cm(y_true, y_pred, filename, figsize=(10,8)):
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
    plt.savefig(filename + '_cm.png')
    return

def plt_barchart(df, title, filename):
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
  plt.savefig(filename + '_bars.png')
  return

def prt_counts(df, title):
  df2 = df.groupby('PriorityID').agg({'PriorityID':['count']}).reset_index()
  df2.columns = ['PriorityID', 'count']
  print(title)
  print("-"*20)
  print(df2)
  print("-"*20)
  total = df2["count"].sum()
  print("Total count = ", total)
  print()
  return

def plot_history(history, string, filename):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  plt.savefig(filename + '_history.png')
  return

def releaseDF(df):
    '''
    Release dataframe from memory
    '''
    list_df = [df]
    del list_df
    return


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
    ############################################################################
    img_path = args.data[:-11]+ "image"

    dataset = pd.read_csv(filename_in, dtype = {'Notes': str})

    #print(dataset.shape)
    #print(dataset.isnull().values.any())
    #print()
    dataset.isnull().any()

    #print(dataset.isnull().T.any().T.sum()) # Number of rows with NaNs
    #print()
    dataset[pd.isnull(dataset).any(axis=1)]

    dataset = dataset.dropna(how="any")
    dataset.reset_index(inplace=True, drop=True)

    #print(dataset.shape)
    #print(dataset.isnull().values.any())

    #dataset.head(5)

    #dataset.info()

    """# D. Modeling Using LSVC"""

    tfidf = TfidfVectorizer(sublinear_tf=True,
                            min_df=5,
                            norm='l2',
                            encoding='latin-1',
                            ngram_range=(1, 3))

    features = tfidf.fit_transform(dataset.text).toarray()
    labels = dataset.label.values
    ##print(features.shape)

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(features,
                                                                             labels,
                                                                             dataset.index,
                                                                             test_size=0.2,
                                                                             shuffle=True,
                                                                             random_state=273)
    #print("train shape: ", x_train.shape)
    #print("test shape:  ", x_test.shape)

    clf_lsvc = LinearSVC()
    clf_lsvc.fit(x_train, y_train)
    y_pred = clf_lsvc.predict(x_test)

    print(metrics.classification_report(y_test,
                                        y_pred,
                                        target_names=["Low (0)","Normal (1)","High (2)","Critical (3)"],
                                        digits=4))
    plot_cm(y_test, y_pred, img_path)

    print("*"*80)
    print("FINISHED LSVM")
    print("*"*80)

