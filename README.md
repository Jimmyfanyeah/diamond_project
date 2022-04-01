# Multi-class classification (OAO strategy)

## Description
This branch is for multi-class classification of small regions for different classes of inclusions.  
One-against-one strategy: train multiple classifier for each pair of classes among all multi classes.  
Take 4 classes, class A, B, C, and D as example. Six networks will train for 
* A and B
* A and C
* A and D
* B and C
* B and D
* C and D



## Getting Started
### Data prepare (Pre-Processing)  
#### Run `data_gen.py` 
* Cut out small patches with center to be target/object & corresponding masks 
* Resize to fixed size  
* Save corresponding txt files for each diamodn ID
* Data structure shows as follow
```
Data Folder
├── Cloud
│   ├── 11-digit.png
│   └── ...
├── ...
├── Crystal
└── txt_file
    ├── 11-digit.txt
    └── ...
```

#### Run `data_split.py`
* For each class (e.g., Crystal), split diamond IDs to train, val, and test phase, save `train_ids.txt`, `val_ids.txt`, and `test_ids.txt`.
* For one-against-one strategy, regroup dataset to the follow structure. 
    * Each folder save data that will be used in the classifer for a specific pair classes. E.g., data in "Cloud_Crystal" will be used when train model for class Cloud and Crystal.
    * Folder 'multiclass_once' save data for one classifier/model trained on all classes, to compare with OAO strategy.
```
Folder
├── Cloud_Crystal
│   ├── test
│   │   ├── Cloud
│   │   └── Crystal
│   ├── train
│   │   ├── Cloud
│   │   └── Crystal
│   └── val
│       ├── Cloud
│       └── Crystal
├── ...
└── multiclass_once
    ├── test
    │   ├── Cloud
    │   ├── Crystal
    │   ├── Feather
    │   └── Twinning_wisp
    ├── train
    │   └── ...
    └── val
        └── ...
```

### Train a model
* [Network architecture](https://github.com/lukemelas/EfficientNet-PyTorch)  
Find all models in `./efficientnet_pytorch/model.py`
* Adjust parameters in `run_main.sh` and `main.py`
* Train network and print top 1 and top 2 accuracy


### Inference
#### Run `Infer.py`
* Choose model (for OAO strategy) by Line 25
* Test all test data (not only those in the paired two classes)
* Save result into csv files for each model, e.g., `Cloud_Crystal.csv`.  
Each column shows info as below

| Sample index | GT class name | GT class index | Class 1 probability | Class 2 probability |
| --- | ----- | -------- | ------------------- | ------------------- |
| 10360576255_43.png | Crystal | 2 | 0.81 | 0.19 |
| ... | ... | ... | ... | ... |

#### Calculate final class
* Merge probabilities from paired classifers, get final probability vector for each sample
* Method in paper
> Chan, R. H., Kan, K. K., Nikolova, M., & Plemmons, R. J. (2020). A two-stage method for spectral–spatial classification of hyperspectral images. Journal of Mathematical Imaging and Vision, 62(6), 790-807. 


### Other information
* [ReadME template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
