set -x
nohup python3 Apex_codes/main.py  \
            --train_or_test='train'  \
            --gpu_number='0,1'  \
            --img_size=512  \
            --batch_size=8  \
            --n_class=1  \
            --augmentation=1  \
            --Epoch=200  \
            --lr=3.90625e-06  \
            --lr_decay_per_epoch=4  \
            --lr_decay_factor=0.5  \
            --bce_weight=1  \
            --save_epoch=20  \
            --data_path='/media/hdd/css_data/1class/clip512'  \
            --train_id_loc='/media/hdd/css_data/1class/txt512/id_train_clip.txt'  \
            --val_id_loc='/media/hdd/css_data/1class/txt512/id_val_clip.txt'  \
            --save_path='/home/lingjia/Documents/diamond_result/1class/Train'  \
            --model_path='/home/lingjia/Documents/diamond_result/1class/Train/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001/20211123-imgSize512-bceWeight1.0-batchSize8-Epoch200-lr0.001-UNet_checkpoint_latest'  \
            --resume  \
            > /home/lingjia/Documents/diamond_result/1class/log/1123_resume.log 2>&1 &
set +x