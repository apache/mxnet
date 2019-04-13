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

# Run BERT QA model using Java Inference API

In this tutorial, we will walk through the BERT QA model trained by MXNet. 
Users can provide a question with a paragraph contains answer to the model and
the model will be able to find the best answer from the answer paragraph.

Example:
```text
Q: When did BBC Japan start broadcasting?
```

Answer paragraph
```text
BBC Japan was a general entertainment channel, which operated between December 2004 and April 2006.
It ceased operations after its Japanese distributor folded.
```
And it picked up the right one:
```text
A: December 2004
```

## Setup Guide

### Step 1: Download the model

For this tutorial, you can get the model and vocabulary by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the `scala-package/examples/scripts/infer/bert/` folder run:

```bash
./get_bert_data.sh
```

### Step 2: Setup data path of the model

### Setup Datapath and Parameters

The available arguments are as follows:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `--model-path-prefix`           | Folder path with prefix to the model (including json, params). |
| `--model-vocab`                 | Vocabulary path |
| `--model-epoch`                 | Epoch number of the model |
| `--input-question`              | Question that asked to the model |
| `--input-answer`                | Paragraph that contains the answer |
| `--seq-length`                  | Sequence Length of the model (384 by default) |

### Step 3: Run Inference
After the previous steps, you should be able to run the code using the following script that will pass all of the required parameters to the Infer API.

From the `scala-package/examples/scripts/infer/bert/` folder run:

```bash
./run_bert_qa_example.sh --model-path-prefix ../models/static-bert-qa/static_bert_qa \
                         --model-vocab ../models/static-bert-qa/vocab.json \
                         --model-epoch 2
```

## Background

To learn more about how BERT works in MXNet, please follow this [MXNet Gluon tutorial on NLP using BERT](https://medium.com/apache-mxnet/gluon-nlp-bert-6a489bdd3340).

The model was extracted from MXNet GluonNLP with static length settings.

[Download link for the script](https://gluon-nlp.mxnet.io/_downloads/bert.zip)

The original description can be found in the [MXNet GluonNLP model zoo](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html#bert-base-on-squad-1-1).
```bash
python static_finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu 0 --export

```
This script will generate `json` and `param` fles that are the standard MXNet model files.
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
