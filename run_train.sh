#SBATCH --job-name=cls
#SBATCH --nodes=1
#SBATCH --partition=gpu_7d1g
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=../sbatch_output/tmp.out
#SBATCH --nodelist=hpc-gpu008

source /home/lingjia/.bashrc
source activate diamond_cls_torch
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
# name_time="2099-12-12-12-12-12" # for test
name_cls="Cloud-Crystal-Feather-Twinning_wisp"
save_name="/media/hdd/lingjia/hdd_diamond/cls/temp/${name_time}_${name_cls}"
mkdir -p -- "$save_name"
current_time=$(date '+%Y-%m-%d-%H-%M-%S')
log_name="${save_name}/${current_time}_train.log"
printf "Name Time: ${name_time}\nStart Time: ${current_time}\n" >> "${log_name}"
export JOBLIB_TEMP_FOLDER=/media/hdd
python3 main.py     --gpu='2'  \
                    --data='/media/hdd/lingjia/hdd_diamond/cls/data/diamond/exp11'  \
                    --epochs=200  \
                    --batch_size=64  \
                    --image_size=224  \
                    --lr=1e-1  \
                    --print_freq=10  \
                    --save_freq=40  \
                    --arch='resnet101'  \
                    --save_path=${save_name}  \
                    --num_cls=4  \
                    --img_cls=${current_cls} \
                    >> "${log_name}"
printf "End: `date`\n\n\n" >> "${log_name}"
set +x

# efficientnet-b0