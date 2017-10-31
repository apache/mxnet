# How to train model for image classification

This folder contains examples relaleted to gluon, for image classfication, use `image_classification.py` to train network on a particular dataset, for example:

- train a 169-layer densenet on the imagenet dataset with batch size 128 and GPU 0,1,2,3
```bash
python image_classification.py --dataset=imagenet --train-data=/path/to/ILSVRC2012_img_train.rec \
    --val-data=/path/to/ILSVRC2012_img_val.rec --gpus=4 --model=densenet169
```
- train a multiplayer perception on mnist dataset
```bash
python mnist.py 
```
