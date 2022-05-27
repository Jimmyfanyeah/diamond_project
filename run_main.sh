set -x
nohup python3 Apex_codes/main.py  \
            --train_or_test='train'  \
            --gpu_number='0,1'  \
            --img_size=256  \
            --batch_size=32  \
            --n_class=1  \
            --augmentation=1  \
            --Epoch=200  \
            --lr=5e-5  \
            --lr_decay_per_epoch=5  \
            --lr_decay_factor=0.5  \
            --bce_weight=0.5  \
            --save_epoch=20  \
            --data_path='/media/hdd/diamond_data/UNet_seg_1class/clip256'  \
            --train_id_loc='/media/hdd/diamond_data/UNet_seg_1class/txt256/id_train_clip.txt'  \
            --val_id_loc='/media/hdd/diamond_data/UNet_seg_1class/txt256/id_val_clip.txt'  \
            --save_path='/home/lingjia/Documents/diamond_result/UNet_seg_1class/Train'  \
            --model_path=''  \
            > /home/lingjia/Documents/diamond_result/UNet_seg_1class/log/20220329_1class_base_ResNet18.log 2>&1 &
set +x