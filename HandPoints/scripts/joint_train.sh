set -ex

# training parameters
MODEL='humansingle'
NETG='joint'
NORM='batch'
BS=32
LR=0.0001

# visulization and save
DISPLAY_ID=11
PORT=8097
CHECKPOINTS_DIR=./checkpoints/
SAVE_EPOCH=20

# dataset
DATAROOT='./data/points_pca'
NUM_POINTS=512

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=${NITER}+${NITER_DECAY}
NAME_BASE=${DATE}_${MODEL}_${NETG}${NGF}_bs${BS}lr${LR}ep${N_EPOCH}
NAME=''
DISPLAY_ENV=${NAME_BASE}_${NAME}
GPU_ID=0

# command
python ./humansingle_main.py \
  --gpu_ids ${GPU_ID} \
  --display_id ${DISPLAY_ID} \
  --dataroot ${DATAROOT} \
  --name ${NAME_BASE}_${NAME} \
  --batch_size ${BS} \
  --model ${MODEL} \
  --netG ${NETG} \
  --fc_embedding \
  --display_port ${PORT} \
  --display_env ${DISPLAY_ENV} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --num_points ${NUM_POINTS} \
  --norm ${NORM} \
  --pool_size 50 \
  --lr ${LR} \
  --dataset_mode 'joint' \
  --lr_policy 'step' \
  --lr_decay_iters 30 \


