set -ex

# training parameters
MODEL='humansingle'
NETG='pointnet' # pointnet++
NORM='batch'
BS=128
LR=0.0001

# dataset
DATAROOT='../data/points_no_pca' # points_no_pca
NUM_POINTS=512

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=200
NAME_BASE=${DATE}_${MODEL}_${NETG}${NGF}_bs${BS}lr${LR}ep${N_EPOCH}
NAME='20k_step30'
DISPLAY_ENV=${NAME_BASE}_${NAME}
GPU_ID=0,1

# command
python ./main_pointnet.py \
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
  --dataset_mode 'joint' \
  --lr_policy 'step' \
  --lr_decay_iters 30 \

