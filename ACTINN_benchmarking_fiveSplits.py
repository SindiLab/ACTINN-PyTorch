
from __future__ import print_function

# std libs
import os
import sys 
import copy
import time
from tqdm import tqdm
import random
import argparse
import numpy as np
import pandas as pd
from math import log10
from sklearn.metrics import f1_score


#AutoClassify
from ACTINN import Classifier, TransferLearning # for attention, import Attn_Classifier. For OG ACTINN import Classifier
from ACTINN import Scanpy_IO, CSV_IO, ScanpyObj_IO
from ACTINN.utils import *

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

parser = argparse.ArgumentParser()

# classifier options
parser.add_argument('--ClassifierEpochs', type=int, default=50, help='number of epochs to train the classifier, default = 50')

parser.add_argument('--data_type', type=str, default="scanpy object", help='type of train/test data, default="scanpy obj"')
parser.add_argument('--dataset_str', type=str, default="PlaceHolder", help='a nickname or actual name for the data')

parser.add_argument('--data_path', type=str, default="", help='Path where the scanpy object (or CSV file) is stored')
parser.add_argument('--metadata_path', type=str, default=None, help='Path where the metadata is stored (which will be used for merging)')


parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=24)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')

parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--print_frequency', type=int, default=5, help='frequency of training stats printing, default=5')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=1000")
parser.add_argument('--cuda', default = True ,action='store_true', help='enables cuda, default = True')
parser.add_argument('--manualSeed', type=int, default = 0, help='manual seed, default = 0')
parser.add_argument('--tensorboard', default=True ,action='store_true', help='enables tensorboard, default True')
parser.add_argument('--outf', default='./TensorBoard/', help='folder to output training stats for tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model, default=None")
parser.add_argument("--finetune", default=True, type=bool, help="to finetune the pretrained model or not, default=True")
parser.add_argument("--reset_epochs", default=False ,action='store_true', help='whether to start training the pretrained model at 0 or not, default=False')


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

str_to_list = lambda x: [int(xi) for xi in x.split(',')]



def main():
    global opt, model
    opt = parser.parse_args()
    
    # determin the device for torch 
    ## if we are allowed to run things on CUDA
    if opt.cuda and torch.cuda.is_available():
        device = "cuda";
        print('==> Using GPU (CUDA)')
        
    else :
        device = "cpu"
        print('==> Using CPU')
        print('    -> Warning: Using CPUs will yield to slower training time than GPUs')
    
    """
    
    Loading in Dataset: We will load in the normalized count matrix from Tabula Muris data
    ## note: we have normalized the data already and saved as scanpy annotated data format
    
    
    """

    if opt.data_type.lower() == "scanpy object":
        
        print(f"==> Reading Scanpy Object for {opt.dataset_str}: ")
        adata = sc.read_h5ad(opt.data_path)
        if opt.metadata_path != None:
            metadata = pd.read_csv(opt.metadata_path)
            print("    -> Merging metadata with existing ann data")
            try:
                adata.obs = adata.obs.merge(metadata, left_on='barcodes', right_on='barcodes', copy=False, suffixes=('', '_drop'))
                adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.str.endswith('_drop')]]
                adata.obs.index = adata.obs['barcodes']
                
            except:
                print("merging on barcode failed, trying to merge on index instead")
                adata.obs['barcodes_orig'] = adata.obs.index.tolist()
                adata.obs = adata.obs.merge(metadata, left_on='barcodes_orig', right_on='index', copy=False, suffixes=('', '_drop'))
                adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.str.endswith('_drop')]]
                adata.obs.index = adata.obs['index']
            
        train_data_loader, valid_data_loader = ScanpyObj_IO(adata,
                                                        test_no_valid = True, 
                                                        batchSize =opt.batchSize, 
                                                        workers = opt.workers,
                                                        verbose = 1,
                                                        raw_X=True)
        
        # get input output information for the network
        inp_size = [batch[0].shape[1] for _, batch in enumerate(valid_data_loader, 0)][0];

        # get the number of celltypes
        number_of_classes = len(adata.obs['celltypes'].unique());
        print(f"==> Number of classes {number_of_classes}")
    
    elif opt.data_type.lower() == "csv":
        # if we have CSV turned to h5 (pandas dataframe)
        train_path = ""
        train_lab_path = ""

        test_path= ""
        test_lab_path= ""

        train_data_loader, valid_data_loader = CSV_IO(train_path, train_lab_path, test_path, test_lab_path,
                                                batchSize=opt.batchSize,
                                                workers = opt.workers)

        # get input output information for the network
        inp_size = [batch[0].shape[1] for _, batch in enumerate(train_data_loader, 0)][0];
        number_of_classes = 9
        print(number_of_classes)
        
    else:
        raise ValueError("Wrong data type, please provide Scanpy/Seurat object or h5 dataframe")
    
    
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    
    """ 
    Building the classifier model:
    """
    cf_model = Classifier(output_dim = number_of_classes, input_size = inp_size).to(device)
    if opt.pretrained:
        print(f"==> Loading pre-trained model from {opt.pretrained}")
        _, trained_epoch = load_model(cf_model, opt.pretrained)
        print(f"    -> Loaded model was trained for {trained_epoch} epochs")
        
        if opt.finetune:
            cf_model = TransferLearning(pretrained_model=cf_model, output_dim=number_of_classes_fineTuneModel, input_size = inp_size).to(device)
        
        if not opt.reset_epochs:
            print(f"    -> NOT resetting the start epoch to 0 ")
            opt.start_epoch = trained_epoch;
        else:
            print(f"    -> resetting the start epoch to 0 ")
        print("    -> Loaded from a pre-trained model:")
    else:
        # initilize the weights in our model
        cf_model.apply(init_weights)
    
    print(f"Model: {cf_model}")
    print(detailed_count_parameters(cf_model))
    print(f"Total number of *trainable parameters* : {count_parameters(cf_model)}")
    
    cf_criterion = torch.nn.CrossEntropyLoss()

    cf_optimizer = torch.optim.Adam(params=cf_model.parameters(), 
                                    lr=0.0001, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-08, 
                                    weight_decay=0.005, 
                                    amsgrad=False)
    cf_decayRate = 0.95
    cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cf_optimizer, gamma=cf_decayRate)
    print("\n Classifier Model \n")

    
    """
    
    Training as the classifier (Should be done when we are warm-starting the VAE part)
    
    """
    
    def train_classifier(cf_epoch, iteration, batch, cur_iter):  
        cf_optimizer.zero_grad()
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            labels = batch[1]
            batch = batch[0]
        batch_size = batch.size(0)
                       
        features= Variable(batch).to(device)
        true_labels = Variable(labels).to(device)
                
        info = f"\n====> Classifier Cur_iter: [{cur_iter}]: Epoch[{cf_epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "
                    
        # =========== Update the classifier ================                  
        pred_cluster = cf_model(features.float()) 
        loss = cf_criterion(pred_cluster.squeeze(), true_labels)
        loss.backward()  
        cf_optimizer.step()
        
        # decaying the LR 
        if cur_iter % opt.step == 0 and cur_iter != 0:
            cf_lr_scheduler.step() 
            for param_group in cf_optimizer.param_groups:
                print(f"    -> Decayed lr to -> {param_group['lr']}")

        # =========== Printing Network Information ================     
        info += f'Loss: {loss.data.item():.4f} ' 
        
        if cur_iter == 0:
            print("    ->Initial stats:", info)

        if epoch % opt.print_frequency == 0 and iteration == (len(train_data_loader) - 1) :
            print(info)
        
    # TRAIN
    benchmarking_dict = {}
    runtimes = []
    macro_f1s = []
    weighted_f1s = []
    accuracies = []
    for split in range(1, 6):
        # resetting the weights so we can learn from scratch!
        print("==> Resetting model weights for new training:")
        cf_model.apply(init_weights)
        print("==> Resetting learning rate for new training:")
        for param_group in cf_optimizer.param_groups:
            param_group['lr'] = opt.lr
        cur_iter = 0
        start_time = time.time()
        print(f"==> Creating a dataloader from split {split}")
        train_data_loader, valid_data_loader = ScanpyObj_IO(adata,
                                                        test_no_valid = True, 
                                                        batchSize =opt.batchSize, 
                                                        workers = opt.workers,
                                                        verbose = 1,
                                                        split_number =split,
                                                        raw_X=True)
        print("---------------- ")
        print(f"==> Trainig Started for split {split}")
        print(f"    -> lr decaying after every {opt.step} steps")
        print(f"    -> Training stats printed after every {opt.print_frequency} epochs")
        for epoch in tqdm(range(0, opt.ClassifierEpochs + 1), desc="Classifier Training"): 
            #save models
            if epoch % opt.print_frequency == 0 and epoch != 0:
                evaluate_classifier(valid_data_loader, cf_model)
                save_epoch = (epoch//opt.save_iter)*opt.save_iter   

            cf_model.train()
            for iteration, batch in enumerate(train_data_loader, 0):
                    #============train Classifier Only============
                    train_classifier(epoch, iteration, batch, cur_iter);
                    cur_iter += 1

        save_epoch = (epoch//opt.save_iter)*opt.save_iter    

        save_checkpoint_classifier(cf_model, save_epoch, 0, f'{opt.dataset_str}')
        print("==> Final evaluation on validation data: ")
        macro_score, w_score, accuracy = evaluate_classifier(valid_data_loader, cf_model, classification_report=True)
        print(f"==> Total training time {time.time() - start_time}");

        benchmarking_dict[f'{opt.dataset_str}_split_{split}'] = [f"runtime = {time.time() - start_time}", f"macro f1 = {macro_score}", 
                                                                 f"weighted_f1 = {w_score}", f"accuracy: {accuracy}"]
        runtimes.append(time.time() - start_time)
        macro_f1s.append(macro_score)
        weighted_f1s.append(w_score)
        accuracies.append(accuracy)
        
    # save benchmarking dictionary
    print("==> Taking average of each metric: ")
    benchmarking_dict[f'{opt.dataset_str}_average_runtimes'] = np.mean(runtimes)
    benchmarking_dict[f'{opt.dataset_str}_average_macro_f1s'] = np.mean(macro_f1s)
    benchmarking_dict[f'{opt.dataset_str}_average_weighted_f1s'] = np.mean(weighted_f1s)
    benchmarking_dict[f'{opt.dataset_str}_average_accuracy'] = np.mean(accuracies)
    print("==> Saving benchmarking dictionary: ")
    Pickler(benchmarking_dict, f"/home/aheydari/data/NACT_Data/Supervised Benchmarking/ACTINN Results/{opt.dataset_str}_fivesplits.pkl")
    
    print(benchmarking_dict)
if __name__ == "__main__":
    main()   
    