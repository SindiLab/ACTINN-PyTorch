import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader

def Scanpy_IO(file_path, batchSize=128, workers = 12):
    """
    Reading in H5AD files that are AnnData object (from Scanpy or Seurat)
    
    INPUTS:
        file_path -> absolute path to the .h5ad file 
        batchSize -> batch size to be used for the PT dataloader
        workers -> number of workers to load/lazy load in data 
    
    RETURN:
        train_data_loader-> training data loader consisting of the data (at batch[0]) and labels (at batch[1])
        test_data_loader-> testing data loader consisting of the data (at batch[0]) and labels (at batch[1])
    
    """
    print("==> Reading in Scanpy/Seurat AnnData")
    adata = sc.read(file_path);

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

    train_data = torch.torch.from_numpy(norm_count_train);
    valid_data = torch.torch.from_numpy(norm_count_valid);

# in case we want to log-scale the data 
#     train_data = torch.log(1 + train_data)
#     valid_data = torch.log(1 + valid_data)

    data_and_labels = []
    validation_data_and_labels = [];
    for i in range(len(train_data)):
        data_and_labels.append([norm_count_train[i], y_train[i]])
        # since validation will always be less than equal to train size
        try:
            validation_data_and_labels.append([norm_count_valid[i], y_valid[i]])
        except:
            pass;

#     print(f"==> sample of the training data: {train_data}");
#     print(f"==> sample of the validation data: {valid_data}");

    inp_size = train_data.shape[1];


    train_data_loader = DataLoader(data_and_labels, batch_size=batchSize, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=workers, collate_fn=None,
           pin_memory=True)

    valid_data_loader = DataLoader(validation_data_and_labels, batch_size=len(valid_data), shuffle=True, sampler=None,
           batch_sampler=None, num_workers=workers, collate_fn=None,
           pin_memory=True)

    return train_data_loader, valid_data_loader
