gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_alternate.py --network pvanet --gpu $1
#python test.py --prefix model/final --epoch 0 --gpu $gpu
