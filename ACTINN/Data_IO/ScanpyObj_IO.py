import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader

def ScanpyObj_IO(obj, test_no_valid:bool = False, batchSize:int = 128, workers:int = 12, log:bool = False, 
              log_base:int = None, log_method: str ='scanpy', verbose = 0, ):
    
    """
    The same function as above, except taking in an AnnData object (from Scanpy or Seurat) and not a path
    
    * The redundency is due to ease of readibility of new users *
    
    INPUTS:
        file_path -> absolute path to the .h5ad file 
        test_or_valid -> boolean to check for test if no validation set is available
        batchSize -> batch size to be used for the PT dataloader
        workers -> number of workers to load/lazy load in data 
        log -> if we want to take log of the data 
        log_base -> which log base we want to use. If None, we will use natural log
        log_method -> if we want to take the log using scanpy or PyTorch
        verbose -> hounding out much printing the code does (not fully used yet)
    
    RETURN:
        train_data_loader-> training data loader consisting of the data (at batch[0]) and labels (at batch[1])
        test_data_loader-> testing data loader consisting of the data (at batch[0]) and labels (at batch[1])
    
    """
    print("==> Validating Scanpy/Seurat Object")
    adata = obj;
    
    if log and log_method == 'scanpy':
        print("    -> Doing log(x+1) transformation with Scanpy")
        sc.pp.log1p(adata, base=log_base)
        
    print("    -> Splitting Train and Validation Data")
    # train
    train_adata = adata[adata.obs['split'].isin(['train'])]
    # validation or test set
    if test_no_valid:
        valid_adata = adata[adata.obs['split'].isin(['test'])]
        
    else:
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
        print("    -> Seems the data is dense")
        norm_count_train = np.asarray(train_adata.X);
        norm_count_valid = np.asarray(valid_adata.X);

    train_data = torch.torch.from_numpy(norm_count_train);
    valid_data = torch.torch.from_numpy(norm_count_valid);

    if log and log_method == "torch":
        print("    -> Doing log(x+1) transformation with torch")
        if log_base == None:
            train_data = torch.log(1 + train_data)
            valid_data = torch.log(1 + valid_data)
        elif log_base == 2:
            train_data = torch.log2(1 + train_data)
            valid_data = torch.log2(1 + valid_data)
        elif log_base == 10:
            train_data = torch.log2(1 + train_data)
            valid_data = torch.log2(1 + valid_data)
        else:
            raise ValueError("    -> We have only implemented log base e, 2 and 10 for torch")
            
    data_and_labels = []
    validation_data_and_labels = [];
    for i in range(len(train_data)):
        data_and_labels.append([norm_count_train[i], y_train[i]])
        # since validation will always be less than equal to train size
        try:
            validation_data_and_labels.append([norm_count_valid[i], y_valid[i]])
        except:
            pass;
    
    if verbose:
        print(f"==> sample of the training data: {train_data}");
        if test_no_valid:
            print(f"==> sample of the test data: {valid_data}");
        else:
            print(f"==> sample of the validation data: {valid_data}");

    inp_size = train_data.shape[1];


    train_data_loader = DataLoader(data_and_labels, batch_size=batchSize, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=workers, collate_fn=None,
           pin_memory=True)

    valid_data_loader = DataLoader(validation_data_and_labels, batch_size=len(valid_data), shuffle=True, sampler=None,
           batch_sampler=None, num_workers=workers, collate_fn=None,
           pin_memory=True)

    return train_data_loader, valid_data_loader