CONFIG=$1
GPUS=$2

PORT=${PORT:-29500}

# Add NCCL timeout & async error handling environment variables
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_freez_backbone.py $CONFIG --launcher pytorch --work-dir "work_dirs/$3"
