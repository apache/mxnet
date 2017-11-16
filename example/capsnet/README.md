**CapsNet-MXNet**
=========================================

This example is MXNet implementation of [CapsNet](https://arxiv.org/abs/1710.09829):  
Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017
- The Current best test error is 0.5%  

Due to the permission issue, this example is maintained in this [repository](https://github.com/samsungsds-rnd/capsnet.mxnet) separately.
* * *
## **Usage**
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

***
## **Results**  
CapsNet classification test error on MNIST

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Train | 0.9281 | 0.9868 | 0.9912 | 0.9932 | 0.9947 | 0.9957 | 0.9965 | 0.9971 | 0.9976 | 0.9980 | 0.9981 | 0.9984 | 0.9986 | 0.9987 | 0.9988 | 0.9989 | 0.9989 |
| Test | 0.9851 | 0.9899 | 0.9919 | 0.9926 | 0.9931 | 0.9933 | 0.9937 | 0.9939 | 0.9943 | 0.9944 | 0.9946 | 0.9945 | 0.9948 | 0.9947 | 0.9948 | 0.9948 | 0.995 |
