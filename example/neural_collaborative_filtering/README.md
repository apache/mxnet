<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Neural Collaborative Filtering

[![Build Status](https://travis-ci.com/xinyu-intel/ncf_mxnet.svg?branch=master)](https://travis-ci.com/xinyu-intel/ncf_mxnet)

This is MXNet implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling. 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

Code Reference: https://github.com/hexiangnan/neural_collaborative_filtering

## Environment Settings
We use MXnet with MKL-DNN as the backend. 
- MXNet version:  '1.5.1'

## Install
```
pip install -r requirements.txt
```

## Dataset

We provide the processed datasets on [Google Drive](https://drive.google.com/drive/folders/1qACR_Zhc2O2W0RrazzcepM2vJeh0MMdO?usp=sharing): MovieLens 20 Million (ml-20m), you can download directly or 
run the script to prepare the datasets:
```
python convert.py ./data/
```

train-ratings.csv
- Train file (positive instances).
- Each Line is a training instance: userID\t itemID\t 

test-ratings.csv
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t 

test-negative.csv
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 999 negative samples.  
- Each line is in the format: userID,\t negativeItemID1\t negativeItemID2 ...

## Pre-trained models

We provide the pretrained ml-20m model on [Google Drive](https://drive.google.com/drive/folders/1qACR_Zhc2O2W0RrazzcepM2vJeh0MMdO?usp=sharing), you can download directly for evaluation or calibration.

|dtype|HR@10|NDCG@10|
|:---:|:--:|:--:|
|float32|0.6393|0.3849|
|int8|0.6366|0.3824|

## Training

```
# train ncf model with ml-20m dataset
python train.py # --gpu=0
```

## Calibration

```
# neumf calibration on ml-20m dataset
python ncf.py --prefix=./model/ml-20m/neumf --calibration
```

## Evaluation

```
# neumf float32 inference on ml-20m dataset
python ncf.py --batch-size=1000 --prefix=./model/ml-20m/neumf
# neumf int8 inference on ml-20m dataset
python ncf.py --batch-size=1000 --prefix=./model/ml-20m/neumf-quantized
```

## Benchmark

```
# neumf float32 benchmark on ml-20m dataset
python ncf.py --batch-size=1000 --prefix=./model/ml-20m/neumf --benchmark
# neumf int8 benchmark on ml-20m dataset
python ncf.py --batch-size=1000 --prefix=./model/ml-20m/neumf-quantized --benchmark
```
