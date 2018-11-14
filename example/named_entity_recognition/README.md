## Goal

- This repo contains an MXNet implementation of this state of the art [entity recognition model](https://www.aclweb.org/anthology/Q16-1026).
- You can find my blog post on the model [here](https://opringle.github.io/2018/02/06/CNNLSTM_entity_recognition.html).

![](https://github.com/dmlc/web-data/blob/master/mxnet/example/ner/arch1.png?raw=true)

## Running the code

To reproduce the preprocessed training data:

1. Download and unzip the data: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv
2. Move ner_dataset.csv into `./data`
3. `$ cd src && python preprocess.py`

To train the model:

- `$ cd src && python ner.py`

To run inference using trained model:

1. Recreate the bucketing module using `sym_gen` defined in `ner.py`
2. Loading saved parameters using `module.set_params()`

Refer to the `test` function in the [Bucketing Module example](https://github.com/apache/incubator-mxnet/blob/master/example/rnn/bucketing/cudnn_rnn_bucketing.py)
and this [issue](https://github.com/apache/incubator-mxnet/issues/5008) on Bucketing Module Prediction