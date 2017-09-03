gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_end2end.py --network pvanet --gpu $1
python test.py --network pvanet --gpu $gpu
