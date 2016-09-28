# -*- coding: utf-8 -*-
"""

"""

import sys
import copy
import os
import numpy as np
import pandas as pd
import scipy as sp
import cPickle as pickle
import random
from sklearn.cross_validation import StratifiedShuffleSplit
import yaml
import csv
from sklearn.metrics import confusion_matrix
import timeit
import shutil
from sklearn.externals import joblib
import gzip

from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_feature_names(feature_groups, features_dict):
    '''
        input: name of feature group key in features dict
        output: flattened list of all feature columns
    '''
    cols = []
    for f in feature_groups:
        if f in features_dict:
            cols.extend(features_dict[f])
        else:
            cols.extend([f])
    return cols



def print_column_names(X):
    for i in X.columns.tolist():
        print( X.columns.tolist().index(i), i)


def print_unique_values(df,column_name):
    for e,i in enumerate(df[column_name].unique()):
        print(e,i)
    print(len(df[column_name].unique()))



def trim_data(x):
    '''
    function to trim low and high values in monthly costs
    '''
    if x < 0:
        x = 0
    elif x > 100000:
        x = 100000
    return x


def truncate_negative(x):
    '''
    function to trim low values in monthly costs
    '''
    if x < 0:
        x = 0
    return x





def get_pearson_corr(y,X,output_file = None, target_name='target'):
    y = pd.Series(np.ravel(y),name=target_name)

    df = pd.concat([y,X],axis=1)
    df.fillna(0,inplace=True)

    pear =df.corr(method='pearson')
    pear = pear.ix[0][1:]
    pear.fillna(0,inplace=True)
    pear = pd.DataFrame(pear)
    # attributes sorted from the most predictive
    pear = pear.sort_values(by=target_name,ascending=False)
    print(pear[:20])
    print(pear[-20:])

    if output_file is not None:
        pear.to_csv(output_file, sep=',')



def print_confusion_matrix(actuals,pred_class):
    actuals= np.ravel(actuals)
    pred_class = np.ravel(pred_class)
    confusion = pd.crosstab(actuals, pred_class , rownames=['True'], colnames=['Predicted'], margins=True)
    confusion = pd.DataFrame(confusion)
    confusion = confusion.iloc[:-1,:-1].div(confusion['All'][:-1],axis=0)
    confusion =confusion.applymap(lambda x: round(100 * x,1))
    print('all values represent percentage of True Class')
    print(confusion)



def split_data(X,y,y_class,test_size=0.25):
    #y_class = str(y_class)
    sss = StratifiedShuffleSplit(y_class, 1, test_size=test_size, random_state= 0)
    for train_index, test_index in sss:
        print("TRAIN:", train_index, "TEST:", test_index)
    test = X.iloc[test_index,:].reset_index(drop=True)
    train = X.iloc[train_index,:].reset_index(drop=True)

    #y=np.array(y)
    y_test = y[test_index].reset_index(drop=True)
    y_train = y[train_index].reset_index(drop=True)
    return train, test, y_train, y_test




##############################################################
# metrics


def get_r_sq(y,pred,naive = None):
    '''
       compute R-squared
    '''
    y = np.ravel(y)
    pred= np.ravel(pred)
    if naive is None:
        naive = np.mean(y)
    return 1-  np.sum((y - pred) **2) /  np.sum( (y -  naive) ** 2 )



def get_r_sq_by_class(y, pred, y_class):
    '''
       compute R-squared by target class
    '''
    y_class = np.ravel(y_class)
    y = np.ravel(y)
    pred = np.ravel(pred)

    errors = {'ClassLabel':[],'R-sq':[],'MeanAbsError':[],'MeanSqError':[]}
    for i in np.unique(y_class):
        label = int(i)
        rsq = get_r_sq(y[y_class==label].copy(),pred[y_class==label].copy(), naive=np.mean(y))
        mean_abs_err = np.mean(np.abs(y[y_class==label] - pred[y_class==label]))
        mean_sq_err = np.sqrt(np.mean(np.square(y[y_class==i] - pred[y_class==i])))

        errors['ClassLabel'].extend([i])
        errors['R-sq'].extend([rsq])
        errors['MeanAbsError'].extend([mean_abs_err])
        errors['MeanSqError'].extend([mean_sq_err])

    print('All',get_r_sq(y, pred),np.mean(np.abs(y- pred)), np.sqrt(np.mean(np.square(y - pred))))

    rsq = get_r_sq(y, pred)
    mean_abs_err = np.mean(np.abs(y- pred))
    mean_sq_err = np.sqrt(np.mean(np.square(y - pred)))

    errors['ClassLabel'].extend(['All'])
    errors['R-sq'].extend([rsq])
    errors['MeanAbsError'].extend([mean_abs_err])
    errors['MeanSqError'].extend([mean_sq_err])

    return pd.DataFrame(errors)


def dump_yaml(dump_obj, path):
    """Save config to disk as yaml file (.yml) in project directory"""
    with open(path, 'w') as yaml_file:
        yaml_file.write(yaml.dump(dump_obj, default_flow_style=False))
        yaml_file.flush()


def load_yaml(load_file):
    """Load config file in yaml format to dictionary

    Parameters
    ----------
    load_file: str
        path to file in yaml format

    Returns
    -------
    dict
        a dicitonary of parameters
    """
    prm = None
    if load_file is not None:
        with open(load_file, "r") as f:
            prm = yaml.safe_load(f)
    return prm


def write_csv(itemlist, filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(itemlist)


def class_accuracy(actual, predicted):
    """Returns a list of class accuracies, order by numeric class label"""
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    conf = confusion_matrix(actual, predicted)
    conf_diag = np.diag(conf)
    class_totals = np.sum(conf, axis=1).astype(float)
    accuracy = conf_diag/class_totals
    accuracy = [round(i, 2) for i in accuracy]
    return accuracy


# Print iterations progress
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def compute_time_delta(start_time, prefix=''):
    """ print or return time (str h-m-s) since start_time

    Parameters
    ----------
    start_time: float
        time

    prefix: str
        prefix to printing time
        output: prefix + time

    print_time: boolean
        if True, then print time to console, else return time as string

    Returns
    -------
    timestr: str
        if print_time is None then return
    """
    stop = timeit.default_timer()
    seconds = stop - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    timestr = "%d:%02d:%02d" % (h, m, s)
    return timestr


def print_time_delta(start_time, prefix=''):
    time_delta = compute_time_delta(start_time, prefix=prefix)
    print(prefix + time_delta)




def clean_dir(path):
    """Remove all files from folder

    :path = ''
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)



def dump(obj, filename):
    """Save object to disk

    This function provides flexibility to use tool of choice for dumping data

    Parameters
    ----------
    obj: any python object

    path: str
        destination dir and file name
    """
    # save object to disk
    joblib.dump(obj, filename)


def load(filename):
    """Load object from disk

    This function provides flexibility to use tool of choice for loading data

    Parameters
    ----------
    obj: any python object

    path: str
        destination dir and file name
    """
    # save object to disk
    return joblib.load(filename)



def gzip_dir(dir_in, dir_out):
    for root, dirs, files in os.walk(dir_in):
        for fname in files:
            filename = os.path.join(root, fname)
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)
            out_file = os.path.join(dir_out, fname + '.gz')
            print(out_file)
            gzip_file(filename, out_file)


def gunzip_dir(dir_in, dir_out):
    for root, dirs, files in os.walk(dir_in):
        for fname in files:
            filename = os.path.join(root, fname)
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)
            out_file = os.path.join(dir_out, fname.replace('.gz', ''))
            print(out_file)
            gunzip_file(filename, out_file)

def gzip_file(f_in, f_out):
    with open(f_in, 'rb') as f_in, gzip.open(f_out, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def gunzip_file(f_in, f_out):
    with gzip.open(f_in, 'rb') as f_in, open(f_out, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def pretty_print_title(title):
    title_len = len(title)
    print('-' * title_len)
    print(title)
    print('-' * title_len)



def align_columns(X, training_features):
    """
    Check for differences in columns between saved dataset and trained model, so that we can reproduce the
    preprocessing and appy the model to a dataset with less columns (aggregated features) For example if we have a
    previously trained model with all companies, and we want to predict for just one company, the saved model has
    to be applied to a dataset that may contain fewer columns, and rarely some entirely new columns. We will
    address the first issue by appending missing columns and filling them with zeros, and the second issue by
    dropping the extra columns.
    _s stands for saved and _c for current


    Parameters
    ----------
    X: pandas dataframe of shape (n_test_samples, n_test_features)

    training_features: list
        features used for training
    """

    columns_s = training_features
    columns_c = list(X.columns.values)
    # print('num columns', len(columns_s), len(columns_c))

    # Drop columns on the processed dataset that are not part of the saved model
    diff_c_s = list(set(columns_c) - set(columns_s))
    if len(diff_c_s) > 0:
        X.drop(diff_c_s, axis=1)

    # Append and fill with zeros columns of the saved model that do not exist in the processed dataset
    df_s = pd.DataFrame(columns=columns_s)
    X = df_s.append(X)
    X.fillna(0, inplace=True)

    # Reorder the columns according to saved model
    X = X[columns_s]

    return X
