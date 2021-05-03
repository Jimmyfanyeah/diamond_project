# diamond project - unet 1 class

## Data Prepare
### process data -> Data_Prepare/dataPrep.py
1. generate mask images
2. delete crust, resize to 1200*1200
3. clip to small patches 400*400
### split to train, val and test set -> gen_txt.py
id_all.txt = ids for train and val
id_val.txt = ids for val

## Training -> run_main.sh
parameters in run_main.sh or main.py
1-class segmentation model trained on origin_UNet -> utils/UNet_Architecture.py, from https://github.com/usuyama/pytorch-unet

## Test
1. run Data_Prepare/dataPrep.py, case='test'
2. test.sh