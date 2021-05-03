set -x
nohup python3 Apex_codes/main.py  \
                    --gpu_number='0,1' \
                    --batchSize=7  \
                    --lr=5e-4  \
                    --lr_decay_factor=0.5 \
                    --model_use='origin_UNet' \
                    --Epoch=5 \
                    --saveEpoch=1000000 \
                    --lr_decay_per_epoch=20 \
                    --data_dir='/home/lingjia/Documents/CSS_Project_1847/hdr_diamonds_labels_clipped_rangefilt' \
                    --val_id_loc='/home/lingjia/Documents/CSS_Project_1847/hdr_diamonds_labels_clipped_rangefilt/id_val.txt' \
                    --id_loc='/home/lingjia/Documents/CSS_Project_1847/hdr_diamonds_labels_clipped_rangefilt/id.txt'  \
                    --model_save_dir='/home/lingjia/Documents/chow/unet1_results/Models' \
                    --save_results_folder='/home/lingjia/Documents/chow/unet1_results/Results' \
                    --save_images_folder='/home/lingjia/Documents/chow/unet1_results/Images' \
                    --resume_training=0 \
                    > 0318_rangefilt.log 2>&1 &
set +x
