import sys
import math
import torch
import argparse
import collections
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader

# Make types to labels dictionary
def type2label_dict(types):
    # input -- types
    # output -- type_to_label dictionary
    type_to_label_dict = {}
    all_type = list(set(types))
    for i in range(len(all_type)):
        type_to_label_dict[all_type[i]] = i
    return type_to_label_dict

# Convert types to labels
def convert_type_to_label(types, type_to_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    types = list(types)
    labels = list()
    for type in types:
        labels.append(type_to_label_dict[type])
    return labels

# Get common genes, normalize  and scale the sets
def scale_sets(sets):
    # input -- a list of all the sets to be scaled
    # output -- scaled sets
    common_genes = set(sets[0].index)
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index),common_genes)
    common_genes = sorted(list(common_genes))
    sep_point = [0]
    for i in range(len(sets)):
        sets[i] = sets[i].loc[common_genes,]
        sep_point.append(sets[i].shape[1])
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
    total_set = np.log2(total_set+1)
    expr = np.sum(total_set, axis=1)
    total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
    total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
    for i in range(len(sets)):
        sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
    return sets


def CSV_IO(train_path, train_labels_path, test_path, batchSize=128, workers = 12):
    print("==> Reading CSV files")
    train_set = pd.read_hdf(train_path, key="dge")
    train_set.index = [s.upper() for s in train_set.index]
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]

    train_label = pd.read_csv(train_labels_path, header=None, sep="\t")

    test_set = pd.read_hdf(test_path, key="dge")
    test_set.index = [s.upper() for s in test_set.index]
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]
    barcode = list(test_set.columns)
    nt = len(set(train_label.iloc[:,1]))

    train_set, test_set = scale_sets([train_set, test_set])
    type_to_label_dict = type2label_dict(train_label.iloc[:,1])
    label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
    print("    -> Cell types in training set:", type_to_label_dict)
    print("    -> # trainng cells:", train_label.shape[0])
    train_label = convert_type_to_label(train_label.iloc[:,1], type_to_label_dict)

    # we want to get Cells X Genes
    train_set = np.transpose(train_set)
    test_set = np.transpose(test_set)

    data_and_labels = []
    validation_data_and_labels = [];
    for i in range(len(train_set)):
        data_and_labels.append([train_set[i], train_label[i]])
        ## will add the test dataloader SOON
        # since test set should be always be less than equal to train size

    # create a DataLoader
    train_data_loader = DataLoader(data_and_labels, batch_size=batchSize, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=workers, collate_fn=None,
           pin_memory=True)

    # SOON 
    # valid_data_loader = DataLoader(validation_data_and_labels, batch_size=len(valid_data), shuffle=True, sampler=None,
    #        batch_sampler=None, num_workers=workers, collate_fn=None,
    #        pin_memory=True)

    return train_data_loader
