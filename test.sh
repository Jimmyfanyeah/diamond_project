
set -x
python3 Apex_codes/main.py   --gpu_number='0' \
                    --batchSize=1  \
                    --lr=1e-4  \
                    --train_or_test='test'  \
                    --data_dir='/home/lingjia/Documents/Diamond_Project_Data/UNET1/diamonds_labels_cutted'  \
                    --id_loc='/home/lingjia/Documents/Diamond_Project_Data/UNET1/id_tmp.txt'  \
                    --save_images_folder='/home/lingjia/Documents/'  \
                    --model_load_dir='/home/lingjia/Documents/chow/unet1_results/Models/2021_01_27-batchSize_7-Epoch_250-lr_0.0001-re_0-UNet_val_dice_best' 
set +x