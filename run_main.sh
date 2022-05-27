set -x
nohup python3 main.py  \
                    --arch='efficientnet-b0'  \
                    --num_cls=1  \
                    --gpu='0,1'  \
                    --image_size=128 \
                    --batch-size=64  \
                    --epochs=200  \
                    --data='/media/hdd/diamond_data/cls_multi-class_EfficientNet_OAO_strategy_groups/Crystal_Feather'  \
                    --save_path='/home/lingjia/Documents/diamond_result/cls_multi-class_EfficientNet/Crystal_Feather'  \
                    > /home/lingjia/Documents/diamond_result/cls_multi-class_EfficientNet/log/Crystal_Feather.log 2>&1 &
set +x