
set -x
python3 Apex_codes/main.py   \
                --train_or_test='test'  \
                --gpu_number='1,2,3' \
                --batch_size=8  \
                --img_size=512 \
                --data_path='/media/hdd/css/frame001/white_cut'  \
                --val_id_loc='/media/hdd/css/frame001/white_cut/id.txt'  \
                --save_path='/media/hdd/css/prediction/1124_all_frame'  \
                --model_path='/home/lingjia/Documents/diamond_result/1class/Train/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001-UNet_val_dice_best' 
set +x

# DATA PATH
# /media/hdd/css_data/1class/cut
# /media/hdd/css/frame_all/10362869233

# /home/lingjia/Documents/diamond_tmp/81_100/id81_100.txt
# /home/lingjia/Documents/diamond_result/1class/unet1_results/Models/2021_01_27-batchSize_7-Epoch_250-lr_0.0001-re_0-UNet_val_dice_best

# MODEL PATH
# /home/lingjia/Documents/diamond_result/unet1_results/Models/20210318-batchSize_7-Epoch_50-lr_5e-06-re_1-UNet_val_dice_best
# /home/lingjia/Documents/diamond_result/unet1_results/Results/20210915-batchSize64-Epoch301-lr0.0001/20210915-batchSize64-Epoch301-lr0.0001-UNet_val_best

# image size=512, loss=bce
# /home/lingjia/Documents/diamond_result/1class/Train/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001-UNet_val_dice_best
# /home/lingjia/Documents/diamond_result/1class/Train/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr3.90625e-06-resume/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr3.90625e-06-resume-UNet_val_dice_best

