**CapsNet-MXNet**
=========================================

This example is MXNet implementation of [CapsNet](https://arxiv.org/abs/1710.09829):  
Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017
- The Current best test error is 0.44%  

Due to the permission issue, this example is maintained in this [repository](https://github.com/samsungsds-rnd/capsnet.mxnet) separately.
* * *
## **Usage**
Install scipy with pip  
```
pip install scipy
```

On Single gpu
```
python capsulenet.py --devices gpu0
```
On Multi gpus
```
python capsulenet.py --devices gpu0,gpu1
```

* * *
## **Prerequisities**

MXNet version above (0.11.0)
scipy version above (0.19.0)

***
## **Results**  
Train time takes about 36 seconds for each epoch (batch_size=100, lr=0.001, 2 gtx 1080 gpus)  
and we limited number of epoch to 100 as default to limit our training time(1 hour).

CapsNet classification test error on MNIST  

| Epoch | train err | test err | train loss | test loss |
| :---: | :---: | :---: | :---: | :---: |
| 40 | 0.23 | 0.44 | 0.00020097 | 0.00022076 |
