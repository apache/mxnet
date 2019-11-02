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


# BERT

There are two examples showcasing the power of BERT. One is BERT-QA for inference and the other is BERT Sentence Pair Classification which uses fine tuning of the BERT base model. For more information about BERT please read [http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/).

## bert-qa

**This example was based off of the Java API one. It shows how to do inference with a pre-trained BERT network that is trained on Questions and Answers using the [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)**

The pretrained model was created using GluonNLP and then exported to the MXNet symbol format. You can find more information in the background section below.

In this tutorial, we will walk through the BERT QA model trained by MXNet. 
Users can provide a question with a paragraph contains answer to the model and
the model will be able to find the best answer from the answer paragraph.

Example:

```
{:input-answer "Steam engines are external combustion engines, where the working fluid is separate from the combustion products. Non-combustion heat sources such as solar power, nuclear power or geothermal energy may be used. The ideal thermodynamic cycle used to analyze this process is called the Rankine cycle. In the cycle, water is heated and transforms into steam within a boiler operating at a high pressure. When expanded through pistons or turbines, mechanical work is done. The reduced-pressure steam is then condensed and pumped back into the boiler."
  :input-question "Along with geothermal and nuclear, what is a notable non-combustion heat source?"
  :ground-truth-answers ["solar"
                         "solar power"
                         "solar power, nuclear power or geothermal energy solar"]}
```

The prediction in this case would be `solar power`

### Setup Guide

Note: If you have trouble with your REPL and cider, please comment out the `lein-jupyter` plugin. There are some conflicts with cider.

#### Step 1: Download the model

For this tutorial, you can get the model and vocabulary by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the example directory:

```bash
./get_bert_data.sh
```

Some sample questions and answers are provide in the `squad-sample.edn` file. Some are taken directly from the SQuAD dataset and one was just made up. Feel free to edit the file and add your own!


### To run

* `lein install` in the root of the main project directory
* cd into this project directory and do `lein run`. This will execute the cpu version.
  * `lein run` or `lein run :cpu` to run with cpu
  * `lein run :gpu` to run with gpu

### Background

To learn more about how BERT works in MXNet, please follow this [MXNet Gluon tutorial on NLP using BERT](https://medium.com/apache-mxnet/gluon-nlp-bert-6a489bdd3340).

The model was extracted from MXNet GluonNLP with static length settings.

[Download link for the script](https://gluon-nlp.mxnet.io/_downloads/bert.zip)

The original description can be found in the [MXNet GluonNLP model zoo](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html#bert-base-on-squad-1-1).
```bash
python static_finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu 0 --export

```
This script will generate `json` and `param` files that are the standard MXNet model files.
By default, this model are using `bert_12_768_12` model with extra layers for QA jobs.

After that, to be able to use it in Java, we need to export the dictionary from the script to parse the text
to actual indexes. Please add the following lines after [this line](https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/staticbert/static_finetune_squad.py#L262).
```python
import json
json_str = vocab.to_json()
f = open("vocab.json", "w")
f.write(json_str)
f.close()
```
This would export the token vocabulary in json format.
Once you have these three files, you will be able to run this example without problems.

## Fine-tuning Sentence Pair Classification with BERT

This was based off of the great tutorial for in Gluon-NLP [https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html](https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html).

We use the pre-trained BERT model that was exported from GluonNLP via the `scripts/bert/staticbert/static_export_base.py` running `python static_export_base.py --seq_length 128`. For convenience, the model has been downloaded for you by running the get_bert_data.sh file in the root directory of this example.

It will fine tune the base bert model for use in a classification task for 3 epochs.


### Setup Guide


## Installation

Before you run this example, make sure that you have the clojure package installed.
In the main clojure package directory, do `lein install`. Then you can run
`lein install` in this directory.

#### Step 1: Download the model

For this tutorial, you can get the model and vocabulary by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the example directory:

```bash
./get_bert_data.sh
```

### To run the notebook walkthrough

There is a Jupyter notebook that uses the `lein jupyter` plugin to be able to execute Clojure code in project setting. The first time that you run it you will need to install the kernel with `lein jupyter install-kernel`. After that you can open the notebook in the project directory with `lein jupyter notebook`.

There is also an exported copy of the walkthrough to markdown `fine-tune-bert.md`.


### To run

* `lein install` in the root of the main project directory
* cd into this project directory and do `lein run`. This will execute the cpu version.

`lein run -m bert.bert-sentence-classification :cpu` - to run with cpu
`lein run -m bert.bert-sentence-classification :gpu` - to run with gpu

By default it will run 3 epochs, you can control the number of epochs with:

`lein run -m bert.bert-sentence-classification :cpu 1` to run just 1 epoch


Sample results from cpu run on OSX
```
INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Train-accuracy=0.65384614
INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Time cost=464187
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [1]	Speed: 0.91 samples/sec	Train-accuracy=0.656250
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [2]	Speed: 0.90 samples/sec	Train-accuracy=0.656250
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [3]	Speed: 0.91 samples/sec	Train-accuracy=0.687500
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [4]	Speed: 0.90 samples/sec	Train-accuracy=0.693750
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [5]	Speed: 0.91 samples/sec	Train-accuracy=0.703125
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [6]	Speed: 0.92 samples/sec	Train-accuracy=0.696429
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [7]	Speed: 0.91 samples/sec	Train-accuracy=0.699219
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [8]	Speed: 0.90 samples/sec	Train-accuracy=0.701389
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [9]	Speed: 0.90 samples/sec	Train-accuracy=0.690625
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [10]	Speed: 0.89 samples/sec	Train-accuracy=0.690341
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [11]	Speed: 0.90 samples/sec	Train-accuracy=0.695313
INFO  org.apache.mxnet.Callback$Speedometer: Epoch[2] Batch [12]	Speed: 0.91 samples/sec	Train-accuracy=0.701923
INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Train-accuracy=0.7019231
INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Time cost=459809
````
