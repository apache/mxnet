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

# cnn-text-classification

An example of text classification using CNN

To use you must download the MR polarity dataset and put it in the path specified in the mr-dataset-path
The dataset can be obtained here: [CNN_sentence](https://github.com/yoonkim/CNN_sentence). The two files `rt-polarity.neg`
and `rt-polarity.pos` must be put in a directory. For example, `data/mr-data/rt-polarity.neg`.

You also must download the glove word embeddings. The suggested one to use is the smaller 50 dimension one
`glove.6B.50d.txt` which is contained in the download file here: [GloVe](https://nlp.stanford.edu/projects/glove/)

## Usage

You can run through the repl with
`(train-convnet {:embedding-size 50 :batch-size 100 :test-size 100 :num-epoch 10 :max-examples 1000 :pretrained-embedding :glove})`

or
`JVM_OPTS="-Xmx1g" lein run` (cpu)

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices


The max-examples only loads 1000 each of the dataset to keep the time and memory down. To run all the examples,
change the main to be (train-convnet {:embedding-size 50 :batch-size 100 :test-size 1000 :num-epoch 10 :pretrained-embedding :glove})

and then run

- `lein uberjar`
- `java -Xms1024m -Xmx2048m -jar target/cnn-text-classification-0.1.0-SNAPSHOT-standalone.jar`

## Usage with word2vec

You can also use word2vec embeddings in order to train the text classification model.
Before training, you will need to download [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) first.
Once you've downloaded the embeddings (which are in a gzipped format),
you'll need to unzip them and place them in the `contrib/clojure-package/data` directory.

Then you can run training on a subset of examples through the repl using:
```
(train-convnet {:embedding-size 300 :batch-size 100 :test-size 100 :num-epoch 10 :max-examples 1000 :pretrained-embedding :word2vec})
```
Note that loading word2vec embeddings consumes memory and takes some time.

You can also train them using `JVM_OPTS="-Xmx8g" lein run` once you've modified
the parameters to `train-convnet` (see above) in `src/cnn_text_classification/classifier.clj`.
In order to run training with word2vec on the complete data set, you will need to run:
```
(train-convnet {:embedding-size 300 :batch-size 100 :test-size 1000 :num-epoch 10 :pretrained-embedding :word2vec})
```
You should be able to achieve an accuracy of `~0.78` using the parameters above.

## Usage with learned embeddings

Lastly, similar to the python CNN text classification example, you can learn the embeddings based on training data.
This can be achieved by setting `:pretrained-embedding nil` (or omitting that parameter altogether).
