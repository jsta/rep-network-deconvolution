# Rep. Network Deconvolution [![DOI](https://img.shields.io/badge/Reproducibility_Study-Network_Deconvolution-blue)](https://github.com/yechengxi/deconvolution)
This repository is a reproducibility study of [Network Deconvolution](https://github.com/yechengxi/deconvolution); the original work proposed by Ye et al. (2020). This reproducibility study is conducted as a part of course requirement for CS895 - Deep Learning Fundamentals course at CS ODU The project requires
Python GPU-based processing capabilities, TensorFlow and
PyTorch frameworks." 

README file of the original study is available at `original_paper/README.md`. 

## Original Paper
```
@inproceedings{
Ye2020Network,
title={Network Deconvolution},
author={Chengxi Ye and Matthew Evanusa and Hua He and Anton Mitrokhin and Tom Goldstein and James A. Yorke and Cornelia Fermuller and Yiannis Aloimonos},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkeu30EtvS }

```

## Folder structure 
```
    .
    ├───documents           # reproducibility study documentation related files
    ├───models              # model architecture implementations
    ├───original_paper      # documents from original study
    ├───results             # results obtained from reproducibility study
    ├───segmentation        # Semantic Segmentation using "Cityscapes" dataset 
    │   arg_parser.py
    │   cs895_project.ipynb
    │   LICENSE
    │   main.py
    │   main_imagenet.py
    │   net_util.py
    │   README.md
    │   requirements.txt
    │   single_experiment_cifar10.sh
    │   single_experiment_cifar100.sh
    │   single_experiment_net_deconv_cifar10.sh
    │   single_experiment_net_deconv_cifar100.sh
    │   util.py
    └   README.md
```


## Dependencies ##
All the required dependencies included in the `requirements.txt` file. Codebase is tested on below python library versions.

* scipy==1.10.1
* numpy==1.23.5
* tensorboard==2.12.0
* matplotlib=3.7.1
* torch==1.13
* torchvision==0.14.1
* tensorflow==2.12.0

## Reproducibility Approach ##
We focused on the reported data from table 1 in the original study for both Batch Normalization and Network Deconvolution scenarios.

![alt text](documents/rep_study_approach.png "Reproduced Results")

## Resolved minor issues ##
There were a few minor module import issues and some Python library version conflicts in the codebase, and below is how we resolved them.
* No module named `torchvision.models.utils`
    - Error with pytorch 1.10+ versions. Resolved by changing 
        ```
        from torchvision.models.utils import load_state_dict_from_url
            to
        from torch.hub import load_state_dict_from_url
        ```
* NameError: name 'DPN92' is not defined
    - A Model import error. Resolved by importing the models to main.py
* NameError: name 'PreActResNet18' is not defined
    - A Model import error. Resolved by importing the models to main.py

## Steps we have followed to reproduce the original study ##

1. Clone the GitHub repository https://github.com/yechengxi/deconvolution
2. Create a python virtual environment https://docs.python.org/3/library/venv.html
3. Activate venv, navigate to the cloned repository and install the dependencies using `requirements.txt` file

    ```
        pip install -r requirements.txt
    ```
4. Created a Jupyter Notebook `cs895_project.ipynb` to test the scripts and 
5. Used bash scripts to schedule slurm jobs and passed related arguments for below parameters:  
    - prameters
        - architecture - neural network architecture name [ vgg16, resnet18d, preact, densenet121, resnext, mobilev2, dpn, pnasnetA, senet, efficient ]
        - epochs - [ 1, 20, 100 ]
        - o -- slurm output fikleename
    - bash commands 
        - for batch normalization with CIFAR-10 dataset [ `single_experiment_cifar10.sh` ]
            ```
            sbatch --export=ALL,architecture='pnasnetA',epochs=100 -o pnasnetA_cifar100_ep100_att2_BN.txt single_experiment_cifar10.sh
            ```
        - for batch normalization with CIFAR-100 dataset [ `single_experiment_cifar100.sh` ]
            ```
            sbatch --export=ALL,architecture='pnasnetA',epochs=100 -o pnasnetA_cifar100_ep100_att2_BN.txt single_experiment_cifar100.sh
            ```
        - for network deconvolution with CIFAR-10 dataset [ `single_experiment_net_deconv_cifar10.sh` ]
            ```
            sbatch --export=ALL,architecture='pnasnetA',epochs=100 -o pnasnetA_cifar100_ep100_att2_BN.txt single_experiment_net_deconv_cifar10.sh
            ```
        - for network deconvolution with CIFAR-100 dataset [ `single_experiment_net_deconv_cifar100.sh` ]
            ```
            sbatch --export=ALL,architecture='pnasnetA',epochs=100 -o pnasnetA_cifar100_ep100_att2_BN.txt single_experiment_net_deconv_cifar100.sh
            ```
6. Stored findings and graphs inside `results` directory 
    - rep_values.xlsx
    - graphs
        - densenet121-ep20-cifar10.png
        - resnet18-ep20-cifar10.png
        - vgg16-ep20-cifar10.png

```BibTeX

```

```
Kumushini Thennakoon | Rochana R. Obadage
04/11/2024
```