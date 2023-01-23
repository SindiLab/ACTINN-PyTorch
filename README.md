# PyTorch implementation of ACTINN

## ACTINN: automated identification of cell types in single cell RNA sequencing 
[Link to paper](https://academic.oup.com/bioinformatics/article-abstract/36/2/533/5540320?redirectedFrom=fulltext)

## Installation
#### Step 1: Install Requirements Explicitly

Ensure that you are in the same directory as `requirements.txt`. Then using `pip`, we can install the requirements with:

````bash
pip install -r requirements.txt
````
Although the core requirements are listed directly in `setup.py`, it is good to run this beforehand in case of any dependecy on packages from GitHub. 

#### Step 2: Install Package Locally
Make sure to be in the same directory as `setup.py`. Then, using `pip`, run:

````bash
pip install -e .
````

For step 2, expect a lot of the requirements to be satisfied already (since you installed the requirements in advance).

## Training Data

We provide two input/output functions (in `Scanpy_IO.py` and `CSV_IO.py`) for compatibility both with the original formatting of ACTINN implementation formating (found [here](https://github.com/mafeiyang/ACTINN)) and Scanpy/Seurat objects. Depending on the type of data, make sure to provide the correct flag for (`--data_type`) either in the code or in bash command calling the main script.

#### Scanpy/AnnData Objects (Preferred)

You can do the standard clustering of the cells using `scanpy`, and the cluster numbers will be used as the training labels. For the training and validation split, randomly add `train` and `valid`/`test` flags to the scanpy object. Then, you can easily pass that object to the `scanpy_IO` function, which returns a train and validation (or test) pytorch dataloader. Here are examples of creating dataloaders from Scanpy object:

To make dataloader objects from a file stored in a path

````python
# to make dataloader objects from a file stored in a path 
from ACTINN import Scanpy_IO

# get training and testing dataloaders
train_data_loader, test_data_loader = Scanpy_IO('PATH/TO/SINGLE-CELL/DATA/file.h5ad',
                                                 batchSize = 128, 
                                                 workers = 32,
                                                 # use this option if there are 'test' samples but not validation
                                                 test_no_valid = True)
````

To make dataloader objects from an existing `scanpy` object in the code:

````python
# to make dataloader objects from a file stored in a path 
from ACTINN import ScanpyObj_IO

# get training and testing dataloaders
train_data_loader, test_data_loader = ScanpyObj_IO(scanpyObj,
                                                   batchSize = 128, 
                                                   workers = 32,
                                                   # use this option if there are 'test' samples but not validation
                                                   test_no_valid = True)
````

#### CSV I/O 

This is the original formatting that ACTINN provides, which includes a CSV files of `Cells x Genes` and a TSV file of the labels. Note that the labels are the cell types. More information about creating the count matrix CSV files and lables TSV can be found on [ACTINN repo](https://github.com/mafeiyang/ACTINN) or [this great benchmarking study](https://github.com/tabdelaal/scRNAseq_Benchmark).


## USAGE
To create the ACTINN classifier network, you can do the following:
````python
import torch
from ACTINN import Classifier

# choose the appropriate device
## e.g.:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the Classifier instance
actinn_model = Classifier(output_dim = number_of_classes, input_size = inp_size).to(device)

# NOW TRAIN JUST LIKE ANY OTHER PYTORCH MODEL
## an example of this is provided in the file called <classify.py> 

````

### Evaluating Classification
We use `sklearn` to evaluate and report the accuracy of the classifications. Here is an example:
````python
from ACTINN.utils import *

# assuming we have a trained model called "actinn_trained"
## assuming we have a test dataloader called "test_data_loader"
### assuming we know which device we want to use, stored in "device"
#### Note: if no device is passed to the following function, it will try to use CUDA if available

evaluate_classifier(test_data_loader, 
                    actinn_trained,
                    classification_report = True,
                    device=None
                    )
````

## Examples
We have provided a full example of classifying the 68K PBMC datasets. The pre-processed data can be downloaded from our S3 bucket [here]().

## Saving and Loading Pre-Trained Models

### Saving ACTINN Models
We provide a function to easily save the model whenever needed. The fuction will create a dictionary: `state = {"epoch": epoch ,"Saved_Model": model}`. `epoch` will store the epoch, and `Saved_Model` will have the actual torch model.
````python
# assuming we have a model that is training called "actinn_training"
curr_epoch = current_epoch; # the current epoch in which we are calling this fucntion
curr_iter = current_iteration; # the current iteration in which we are calling this fucntion

save_checkpoint_classifier(actinn_training, curr_epoch, current_iter, 'SOME-PREFIX IF YOU WANT')
````

### Loading Pre-Trained Models
Our implementation of ACTINN automatically saves the model at last iteration of training. To load in:
````python
import torch 
import ACTINN

model_dict = torch.load("/home/ubuntu/SindiLab/SCIV/ClassifierWeights/pbmc-model_epoch_10_iter_0.pth")

# REMEMBER: 
## we saved the epoch number and the model in a dictionary -> state = {"epoch": epoch ,"Saved_Model": model}
actinn = model_dict["Saved_Model"]

## evaluate or use just like any other pytorch model
actinn.eval();
````

***Please Cite the following if you use this package***

## Citation

If this implementation was useful for your research, please cite our paper (in which we introduce this implementaton):

```
@article{HeydariEtAl,
author = {Heydari, A. Ali and Davalos, Oscar A and Zhao, Lihong and Hoyer, Katrina K and Sindi, Suzanne S},
date-added = {2023-01-23 12:50:22 -0800},
date-modified = {2023-01-23 12:50:22 -0800},
doi = {10.1093/bioinformatics/btac095},
eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2194/43370117/btac095\_supplementary\_data.pdf},
issn = {1367-4803},
journal = {Bioinformatics},
month = {02},
number = {8},
pages = {2194-2201},
title = {{ACTIVA: realistic single-cell RNA-seq generation with automatic cell-type identification using introspective variational autoencoders}},
url = {https://doi.org/10.1093/bioinformatics/btac095},
volume = {38},
year = {2022},
bdsk-url-1 = {https://doi.org/10.1093/bioinformatics/btac095}}
```

and the original paper:
```
@article{10.1093/bioinformatics/btz592,
    author = {Ma, Feiyang and Pellegrini, Matteo},
    title = "{ACTINN: automated identification of cell types in single cell RNA sequencing}",
    journal = {Bioinformatics},
    volume = {36},
    number = {2},
    pages = {533-538},
    year = {2019},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz592},
    url = {https://doi.org/10.1093/bioinformatics/btz592},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/2/533/31962865/btz592.pdf},
}
```


