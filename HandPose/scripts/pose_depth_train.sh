set -ex

# training parameters
BS=64
LR=0.0001

# dataset
DATAROOT='../data/depth_pose' # points_no_pca
PREPROCESS='rotate_jitter' #resize_and_crop None
#PREPROCESS='none' #resize_and_crop None

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=200
GPU_ID=0,1
NAME_BASE=${DATE}_bs${BS}lr${LR}ep${N_EPOCH}_gpu${GPU_ID}_${PREPROCESS}
NAME=''
DISPLAY_ENV=${NAME_BASE}_${NAME}

# command
python ./main_depth.py \
  --gpu_ids ${GPU_ID} \
  --preprocess ${PREPROCESS} \
  --dataroot ${DATAROOT} \
  --name ${DISPLAY_ENV} \
  --display_env ${DISPLAY_ENV} \
  --batch_size ${BS} \
  --lr ${LR} \
  --dataset_mode 'depth_pose' \
  --lr_policy 'step' \
  --lr_decay_iters 50 \

