from __future__ import print_function

# std libs
import os
import sys 
import copy
import time
import random
import argparse
import numpy as np
from math import log10

#AutoClassify
from AutoClassify import Classifier
from statistics import mean, stdev 
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold 
from sklearn.neural_network import MLPClassifier
from sklearn import datasets 

# reading in single cell data using scanpy
import scanpy as sc


# torch libs
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
## in a future release vv
from tensorboardX import SummaryWriter

# anamoly detection
torch.autograd.set_detect_anomaly(True)



print("==> Reading in Data")
adata = sc.read('/home/jovyan/68K_PBMC_scGAN_Process/raw_68kPBMCs.h5ad');

print("    ->Splitting Train and Validation Data")
# train
train_adata = adata[adata.obs['split'].isin(['train'])]
# validation
valid_adata = adata[adata.obs['split'].isin(['valid'])]

# turn the cluster numbers into labels 
print("==> Using cluster info for generating train and validation labels")
y_train = [int(x) for x in train_adata.obs['cluster'].to_list()]
y_valid = [int(x) for x in valid_adata.obs['cluster'].to_list()]

print("==> Checking if we have sparse matrix into dense")
try:
    norm_count_train = np.asarray(train_adata.X.todense());
    norm_count_valid = np.asarray(valid_adata.X.todense());
except:
    print("    ->Seems the data is dense")
    norm_count_train = np.asarray(train_adata.X);
    norm_count_valid = np.asarray(valid_adata.X);


# Input_x_Features. 
x = norm_count_train[0:10000]					 

# Input_ y_Target_Variable. 
y = y_train[0:10000]			

# Feature Scaling for input features. 
scaler = preprocessing.MinMaxScaler() 
x_scaled = scaler.fit_transform(x) 

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 25, 11), random_state=1)

# Create StratifiedKFold object. 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) 
lst_accu_stratified = [] 

for train_index, test_index in skf.split(x, y): 
	x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index] 
	y_train_fold, y_test_fold = y[train_index], y[test_index] 
	clf.fit(x_train_fold, y_train_fold) 
	lst_accu_stratified.append(clf.score(x_test_fold, y_test_fold)) 

# Print the output. 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
	max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
	min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
	mean(lst_accu_stratified)*100, '%') 
print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# # This code may not be run on GFG IDE 
# # as required packages are not found. 
	
# # STRATIFIES K-FOLD CROSS VALIDATION { 10-fold } 

# # Import Required Modules. 
# from statistics import mean, stdev 
# from sklearn import preprocessing 
# from sklearn.model_selection import StratifiedKFold 
# from sklearn.neural_network import MLPClassifier
# from sklearn import datasets 

# # FEATCHING FEATURES AND TARGET VARIABLES IN ARRAY FORMAT. 
# cancer = datasets.load_breast_cancer() 
# # Input_x_Features. 
# x = cancer.data						 

# # Input_ y_Target_Variable. 
# y = cancer.target					 
	

# # Feature Scaling for input features. 
# scaler = preprocessing.MinMaxScaler() 
# x_scaled = scaler.fit_transform(x) 

# # Create classifier object. 
# lr = linear_model.LogisticRegression() 

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 25, 11), random_state=1)

# # Create StratifiedKFold object. 
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) 
# lst_accu_stratified = [] 

# for train_index, test_index in skf.split(x, y): 
# 	x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index] 
# 	y_train_fold, y_test_fold = y[train_index], y[test_index] 
# 	lr.fit(x_train_fold, y_train_fold) 
# 	lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold)) 

# # Print the output. 
# print('List of possible accuracy:', lst_accu_stratified) 
# print('\nMaximum Accuracy That can be obtained from this model is:', 
# 	max(lst_accu_stratified)*100, '%') 
# print('\nMinimum Accuracy:', 
# 	min(lst_accu_stratified)*100, '%') 
# print('\nOverall Accuracy:', 
# 	mean(lst_accu_stratified)*100, '%') 
# print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 

