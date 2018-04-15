## Goal

- This repo contains an MXNet implementation of this state of the art [entity recognition model](https://www.aclweb.org/anthology/Q16-1026).
- You can find my blog post on the model [here](https://opringle.github.io/2018/02/06/CNNLSTM_entity_recognition.html).

![](https://github.com/dmlc/web-data/blob/master/mxnet/example/ner/arch1.png?raw=true)

## Running the code

To reproduce the preprocessed training data:

1. Download and unzip the data: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv
2. Move ner_dataset.csv into `./data`
3. create `./preprocessed_data` directory
3. `$ cd src && python preprocess.py`

To train the model:

- `$ cd src && python ner.py`