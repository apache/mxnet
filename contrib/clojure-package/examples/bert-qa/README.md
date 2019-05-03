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


# bert-qa

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
                         "solar power, nuclear power or geothermal energysolar"]}
```

The prediction in this case would be `solar power`

## Setup Guide

### Step 1: Download the model

For this tutorial, you can get the model and vocabulary by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the example directory:

```bash
./get_bert_data.sh
```

Some sample questions and answers are provide in the `squad-sample.edn` file. Some are taken directly from the SQuAD dataset and one was just made up. Feel free to edit the file and add your own!


## To run

* `lein install` in the root of the main project directory
* cd into this project directory and do `lein run`. This will execute the cpu version.
  * `lein run` or `lein run :cpu` to run with cpu
  * `lein run :gpu` to run with gpu

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

