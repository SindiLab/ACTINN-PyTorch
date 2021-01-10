# PyTorch implementation of ACTINN

## ACTING: automated identification of cell types in single cell RNA sequencing 
[Link to paper](https://academic.oup.com/bioinformatics/article-abstract/36/2/533/5540320?redirectedFrom=fulltext)

## Installation
To install the package, clone the repository. Then, run 

````bash
pip install -e PATH/TO/FOLDER/WITH-setup.py
````
Since `setup.py` is a fairly standard setup file, other methods of installing the package will work as well (as long as tried locally). 

## Training Data

We provide two input/output functions (in `Scanpy_IO.py` and `CSV_IO.py`) for compatibility both with the original formatting of ACTINN implementation formating (found [here](https://github.com/mafeiyang/ACTINN)) and Scanpy/Seurat objects. Depending on the type of data, make sure to provide the correct flag for (`--data_type`) either in the code or in bash command calling the main script.

#### Scanpy/AnnData Objects (Preferred)

You can do the standard clustering of the cells using `scanpy`, and the cluster numbers will be used as the training labels. For the training and validation split, randomly add `train` and `valid`/`test` flags to the scanpy object. Then, you can easily pass that object to the `scanpy_IO` function, which returns a train and validation (or test) pytorch dataloader. 

#### CSV I/O 

This is the original formatting that ACTINN provides, which includes a CSV files of `Cells x Genes` and a TSV file of the labels. Note that the labels are the cell types. More information about creating the count matrix CSV files and lables TSV can be found on [ACTINN repo](https://github.com/mafeiyang/ACTINN) or [this great benchmarking study](https://github.com/tabdelaal/scRNAseq_Benchmark).


## USAGE


## Examples
We have provided a full example of classifying the 68K PBMC datasets. The pre-processed data can be downloaded from our S3 bucket [here]().

## Saving and Loading Pre-Trained Models


***Please Cite the following if you use this package***

## Citation

Please cite our repository if it was useful for your research:

```
@misc{Heydari2020,
  author = {Heydari, A. Ali},
  title = {PyTorch implementation of ACTINN},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SindiLab/ACTINN-PyTorch}},
}
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


