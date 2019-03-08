# SNAIL with Gluon

---

Gluon inplementation of [A Simple Neural Attentive Meta-Learniner](https://openreview.net/pdf?id=B1DmUzWAW)

##### network structore
![net_structure](assets/net_structure.png)

##### building block structure
![block_structure](assets/blocks.png)

## Requirements
- Python 3.6.1
- mxnet 1.3.1
- mxboard 0.1.0
- tqdm 4.29.0


## Application
-  Omniglot

## Usage

- arguments
  - batch_size : Define batch size (defualt=64)
  - epochs : Define total epoches (default=50)
  - N : the nunber of N-way (default=10)
  - K : the number of K-shot (default=5)
  - iterations : the number of data iteration (default=1000)
  - input_dims : embedding dimension of input data (default=64)
  - download : download omniglot dataset (default=False)
  - GPU_COUNT : use gpu count  (default=1)
  - logdir : location of mxboard log file (default=./log)
  - modeldir : location of model parameter file (default=./models)


###### default setting
```
python main.py
``` 
or

###### manual setting
```
python main.py --batch_size=24 --epochs=200 ..
```

## Results
##### 10-way 5-shot case
![perf_acc](assets/perf_acc.png)


## Reference
- https://github.com/sagelywizard/snail
- https://github.com/eambutu/snail-pytorch

