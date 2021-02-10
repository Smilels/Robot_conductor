set -ex

# training parameters
NORM='batch'
BS=32
LR=0.0001

# dataset
DATAROOT='../data/points_keypoints' # points_no_pca
DATAMODE='keypoints' # joint
NUM_POINTS=1024

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=250
GPU_ID=2,3
NAME_BASE=${DATE}_${MODEL}_${NETG}${NGF}${NUM_POINTS}_bs${BS}lr${LR}ep${N_EPOCH}_gpu${GPU_ID}
NAME='20k'
DISPLAY_ENV=${NAME_BASE}_${NAME}

# command
python ./main_pct_keypoints.py \
  --gpu_ids ${GPU_ID} \
  --dataroot ${DATAROOT} \
  --name ${DISPLAY_ENV} \
  --display_env ${DISPLAY_ENV} \
  --batch_size ${BS} \
  --model ${MODEL} \
  --netG ${NETG} \
  --num_points ${NUM_POINTS} \
  --norm ${NORM} \
  --lr ${LR} \
  --dataset_mode ${DATAMODE} \
  --n_epochs ${N_EPOCH} \
