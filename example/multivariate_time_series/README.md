# Implementation

This tutorial shows how to implement LSTNet, a multivariate time series forecasting model submitted by Wei-Cheng Chang, Yiming Yang, Hanxiao Liu and Guokun Lai in their paper [Modeling Long- and Short-Term Temporal Patterns](https://arxiv.org/pdf/1703.07015.pdf) in March 2017.  This model achieved state of the art performance on 3 of the 4 public datasets it was evaluated on.

## Running the code

1. Download and unpack the public electricity dataset used in the paper.  This dataset comprises measurements of electricity consumption in kWh every hour from 2012 to 2014 for 321 different clients.

```s
$ wget https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz
$ gunzip electricity.txt.gz
```

2. preprocess data with `python preprocess.py`
3. set model hyperparameters in `/src/config.py`
4. `python train.py`