set -ex

# training parameters
BS=128
LR=0.0001

# dataset
DATAROOT='../data/depth_pose' # points_no_pca
TRAIN_NAME='train_20k'
TEST_NAME='test_20k'
PREPROCESS='rotate_jitter' #resize_and_crop None
#PREPROCESS='none' #resize_and_crop None

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=200
GPU_ID=2,3
NAME_BASE=${DATE}_bs${BS}lr${LR}ep${N_EPOCH}_gpu${GPU_ID}_${TRAIN_NAME}${PREPROCESS}
NAME='samelossweight'
DISPLAY_ENV=${NAME_BASE}_${NAME}

# command
python ./main_depth.py \
  --gpu_ids ${GPU_ID} \
  --preprocess ${PREPROCESS} \
  --dataroot ${DATAROOT} \
  --dataset_mode 'depthpose' \
  --name ${DISPLAY_ENV} \
  --display_env ${DISPLAY_ENV} \
  --batch_size ${BS} \
  --lr ${LR} \
  --lr_policy 'step' \
  --lr_decay_iters 50 \
  --train_name ${TRAIN_NAME} \
  --test_name ${TEST_NAME} \
  --load_size 96 \

