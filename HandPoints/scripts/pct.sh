set -ex

# training parameters
MODEL='pct' # 'humansingle'
NETG='pct' # pointnet++
NORM='batch'
BS=32
LR=0.0001

# dataset
DATAROOT='../data/points_no_pca' # points_no_pca
NUM_POINTS=1024

# naming
DATE=`date '+%Y%m%d%H'`
N_EPOCH=300
GPU_ID=2,3
NAME_BASE=${DATE}_${MODEL}_${NETG}${NGF}${NUM_POINTS}_bs${BS}lr${LR}ep${N_EPOCH}_gpu${GPU_ID}
NAME='20k'
DISPLAY_ENV=${NAME_BASE}_${NAME}

# command
python ./main_pct.py \
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
