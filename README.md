# One-class segmentation update 2021/10/18

## Description
This branch is for one class segmentation. All types of inclusions and reflections are considered as one class. The aim is to find out all regions.  

## Getting Started
### Data prepare (Pre-Processing)  
#### In `dataPrep.py` 
* Generate masks  
* Delete black crust and resize to given size (optional)  
* Cut iamges and masks to small patches (64*64) with stride 32. Two ways to cut,
    * cut randomly  
    * cut with the center being a defect  

#### In `helper.py`
* Count number of examples (small patchs) with and without targets
* Split IDs to train, val, test set, save txt file for both cut and clip folder
* Given ratio = with : without target, pick out examples as training set, the pick-out examples are saved in `clip/id_train_adjust.txt` and `clip/id_val_adjust.txt`
* Check mean and std for train dataset

### Train a model
* [Network architecture](https://github.com/usuyama/pytorch-unet)  
* Find optimal learning rate
    * Modify the way of return variables in ```utils/data.py```.
    * Run ```lr_find.py``` before train, to get the optimal learing rate.  
* Result save in `diamond_result/one_class_seg/result`


### Other information
* [ReadME template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
