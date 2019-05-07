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


# Fine-tuning Sentence Pair Classification with BERT

**This tutorial is based off of the Gluon NLP one here https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html**

Pre-trained language representations have been shown to improve many downstream NLP tasks such as question answering, and natural language inference. To apply pre-trained representations to these tasks, there are two strategies:

feature-based approach, which uses the pre-trained representations as additional features to the downstream task.
fine-tuning based approach, which trains the downstream tasks by fine-tuning pre-trained parameters.
While feature-based approaches such as ELMo [3] (introduced in the previous tutorial) are effective in improving many downstream tasks, they require task-specific architectures. Devlin, Jacob, et al proposed BERT [1] (Bidirectional Encoder Representations from Transformers), which fine-tunes deep bidirectional representations on a wide range of tasks with minimal task-specific parameters, and obtained state- of-the-art results.

In this tutorial, we will focus on fine-tuning with the pre-trained BERT model to classify semantically equivalent sentence pairs. Specifically, we will:

load the state-of-the-art pre-trained BERT model and attach an additional layer for classification,
process and transform sentence pair data for the task at hand, and
fine-tune BERT model for sentence classification.



## Preparation

To run this tutorial locally, in the example directory:

1. Get the model and supporting data by running `get_bert_data.sh`. 
2. This Jupyter Notebook uses the lein-jupyter plugin to be able to execute Clojure code in project setting. The first time that you run it you will need to install the kernel with`lein jupyter install-kernel`. After that you can open the notebook in the project directory with `lein jupyter notebook`.

## Load requirements

We need to load up all the namespace requires


```clojure
(ns bert.bert-sentence-classification
  (:require [bert.util :as bert-util]
            [clojure-csv.core :as csv]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]))

```

# Use the Pre-trained BERT Model

In this tutorial we will use the pre-trained BERT model that was exported from GluonNLP via the `scripts/bert/staticbert/static_export_base.py`. For convenience, the model has been downloaded for you by running  the `get_bert_data.sh` file in the root directory of this example.

## Get BERT

Let’s first take a look at the BERT model architecture for sentence pair classification below:

![bert](https://gluon-nlp.mxnet.io/_images/bert-sentence-pair.png)

where the model takes a pair of sequences and pools the representation of the first token in the sequence. Note that the original BERT model was trained for masked language model and next sentence prediction tasks, which includes layers for language model decoding and classification. These layers will not be used for fine-tuning sentence pair classification.

Let's load the pre-trained BERT using the module API in MXNet.


```clojure
(def model-path-prefix "data/static_bert_base_net")
;; the vocabulary used in the model
(def vocab (bert-util/get-vocab))
;; the input question
;; the maximum length of the sequence
(def seq-length 128)

(def bert-base (m/load-checkpoint {:prefix model-path-prefix :epoch 0}))
```




    #'bert.bert-sentence-classification/bert-base



## Model Definition for Sentence Pair Classification

Now that we have loaded the BERT model, we only need to attach an additional layer for classification. We can do this by defining a fine tune model from the symbol of the base BERT model.


```clojure
(defn fine-tune-model
  "msymbol: the pretrained network symbol
   num-classes: the number of classes for the fine-tune datasets
   dropout: the dropout rate"
  [msymbol {:keys [num-classes dropout]}]
  (as-> msymbol data
    (sym/dropout {:data data :p dropout})
    (sym/fully-connected "fc-finetune" {:data data :num-hidden num-classes})
    (sym/softmax-output "softmax" {:data data})))

(def model-sym (fine-tune-model (m/symbol bert-base) {:num-classes 2 :dropout 0.1}))
```




    #'bert.bert-sentence-classification/model-sym



# Data Preprocessing for BERT

## Dataset

For demonstration purpose, we use the dev set of the Microsoft Research Paraphrase Corpus dataset. The file is named ‘dev.tsv’ and was downloaded as part of the data script. Let’s take a look at the raw dataset.


```clojure
(-> (sh "head" "-n" "5" "data/dev.tsv") 
    :out
    println)
```

    ﻿Quality	#1 ID	#2 ID	#1 String	#2 String
    1	1355540	1355592	He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .	" The foodservice pie business does not fit our long-term growth strategy .
    0	2029631	2029565	Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war .	His wife said he was " 100 percent behind George Bush " and looked forward to using his years of training in the war .
    0	487993	487952	The dollar was at 116.92 yen against the yen , flat on the session , and at 1.2891 against the Swiss franc , also flat .	The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .
    1	1989515	1989458	The AFL-CIO is waiting until October to decide if it will endorse a candidate .	The AFL-CIO announced Wednesday that it will decide in October whether to endorse a candidate before the primaries .
    


The file contains 5 columns, separated by tabs (i.e. ‘

\t
‘). The first line of the file explains each of these columns: 0. the label indicating whether the two sentences are semantically equivalent 1. the id of the first sentence in this sample 2. the id of the second sentence in this sample 3. the content of the first sentence 4. the content of the second sentence

For our task, we are interested in the 0th, 3rd and 4th columns. 


```clojure
(def raw-file 
    (csv/parse-csv (string/replace (slurp "data/dev.tsv") "\"" "")
                   :delimiter \tab
                   :strict true))

(def data-train-raw (->> raw-file
                         (mapv #(vals (select-keys % [3 4 0])))
                         (rest) ; drop header
                         (into [])))

(def sample (first data-train-raw))
(println (nth sample 0)) ;;;sentence a
(println (nth sample 1)) ;; sentence b
(println (nth sample 2)) ;; 1 means equivalent, 0 means not equivalent
```

    He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .
     The foodservice pie business does not fit our long-term growth strategy .
    1


To use the pre-trained BERT model, we need to preprocess the data in the same way it was trained. The following figure shows the input representation in BERT:

![bert-input](https://gluon-nlp.mxnet.io/_images/bert-embed.png)

We will do pre-processing on the inputs to get them in the right format and to perform the following transformations:
- tokenize the input sequences
- insert [CLS] at the beginning
- insert [SEP] between sentence one and sentence two, and at the end - generate segment ids to indicate whether a token belongs to the first sequence or the second sequence.
- generate valid length


```clojure
(defn pre-processing
  "Preprocesses the sentences in the format that BERT is expecting"
  [ctx idx->token token->idx train-item]
    (let [[sentence-a sentence-b label] train-item
       ;;; pre-processing tokenize sentence
          token-1 (bert-util/tokenize (string/lower-case sentence-a))
          token-2 (bert-util/tokenize (string/lower-case sentence-b))
          valid-length (+ (count token-1) (count token-2))
        ;;; generate token types [0000...1111...0000]
          qa-embedded (into (bert-util/pad [] 0 (count token-1))
                            (bert-util/pad [] 1 (count token-2)))
          token-types (bert-util/pad qa-embedded 0 seq-length)
        ;;; make BERT pre-processing standard
          token-2 (conj token-2 "[SEP]")
          token-1 (into [] (concat ["[CLS]"] token-1 ["[SEP]"] token-2))
          tokens (bert-util/pad token-1 "[PAD]" seq-length)
        ;;; pre-processing - token to index translation
          indexes (bert-util/tokens->idxs token->idx tokens)]
    {:input-batch [indexes
                   token-types
                   [valid-length]]
     :label (if (= "0" label)
              [0]
              [1])
     :tokens tokens
     :train-item train-item}))

(def idx->token (:idx->token vocab))
(def token->idx (:token->idx vocab))
(def dev (context/default-context))
(def processed-datas (mapv #(pre-processing dev idx->token token->idx %) data-train-raw))
(def train-count (count processed-datas))
(println "Train Count is = " train-count)
(println "[PAD] token id = " (get token->idx "[PAD]"))
(println "[CLS] token id = " (get token->idx "[CLS]"))
(println "[SEP] token id = " (get token->idx "[SEP]"))
(println "token ids = \n"(-> (first processed-datas) :input-batch first)) 
(println "segment ids = \n"(-> (first processed-datas) :input-batch second)) 
(println "valid length = \n" (-> (first processed-datas) :input-batch last)) 
(println "label = \n" (-> (second processed-datas) :label)) 


```

    Train Count is =  408
    [PAD] token id =  1
    [CLS] token id =  2
    [SEP] token id =  3
    token ids = 
     [2 2002 2056 1996 0 11345 2449 2987 0 4906 1996 2194 0 0 3930 5656 0 1012 3 0 1996 0 11345 2449 2515 2025 4906 2256 0 3930 5656 0 1012 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    segment ids = 
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    valid length = 
     [31]
    label = 
     [0]


Now that we have all the input-batches for each row, we are going to slice them up column-wise and create NDArray Iterators that we can use in training


```clojure
(defn slice-inputs-data
  "Each sentence pair had to be processed as a row. This breaks all
  the rows up into a column for creating a NDArray"
  [processed-datas n]
  (->> processed-datas
       (mapv #(nth (:input-batch %) n))
       (flatten)
       (into [])))

(def prepared-data {:data0s (slice-inputs-data processed-datas 0)
                    :data1s (slice-inputs-data processed-datas 1)
                    :data2s (slice-inputs-data processed-datas 2)
                    :labels (->> (mapv :label processed-datas)
                                 (flatten)
                                 (into []))
                    :train-num (count processed-datas)})

(def batch-size 32)

(def train-data
  (let [{:keys [data0s data1s data2s labels train-num]} prepared-data
        data-desc0 (mx-io/data-desc {:name "data0"
                                     :shape [train-num seq-length]
                                     :dtype dtype/FLOAT32
                                     :layout layout/NT})
        data-desc1 (mx-io/data-desc {:name "data1"
                                     :shape [train-num seq-length]
                                     :dtype dtype/FLOAT32
                                     :layout layout/NT})
        data-desc2 (mx-io/data-desc {:name "data2"
                                     :shape [train-num]
                                     :dtype dtype/FLOAT32
                                     :layout layout/N})
        label-desc (mx-io/data-desc {:name "softmax_label"
                                     :shape [train-num]
                                     :dtype dtype/FLOAT32
                                     :layout layout/N})]
    (mx-io/ndarray-iter {data-desc0 (ndarray/array data0s [train-num seq-length]
                                                   {:ctx dev})
                         data-desc1 (ndarray/array data1s [train-num seq-length]
                                                   {:ctx dev})
                         data-desc2 (ndarray/array data2s [train-num]
                                                   {:ctx dev})}
                        {:label {label-desc (ndarray/array labels [train-num]
                                                           {:ctx dev})}
                         :data-batch-size batch-size})))
train-data
```




    #object[org.apache.mxnet.io.NDArrayIter 0x2583097d "non-empty iterator"]



# Fine-tune BERT Model

Putting everything together, now we can fine-tune the model with a few epochs. For demonstration, we use a fixed learning rate and skip validation steps.


```clojure
(def num-epoch 3)

(def fine-tune-model (m/module model-sym {:contexts [dev]
                                         :data-names ["data0" "data1" "data2"]}))

(m/fit fine-tune-model {:train-data train-data  :num-epoch num-epoch
                        :fit-params (m/fit-params {:allow-missing true
                                                   :arg-params (m/arg-params bert-base)
                                                   :aux-params (m/aux-params bert-base)
                                                   :optimizer (optimizer/adam {:learning-rate 5e-6 :episilon 1e-9})
                                                   :batch-end-callback (callback/speedometer batch-size 1)})})

```

    Speedometer: epoch  0  count  1  metric  [accuracy 0.609375]
    Speedometer: epoch  0  count  2  metric  [accuracy 0.6041667]
    Speedometer: epoch  0  count  3  metric  [accuracy 0.5703125]
    Speedometer: epoch  0  count  4  metric  [accuracy 0.55625]
    Speedometer: epoch  0  count  5  metric  [accuracy 0.5625]
    Speedometer: epoch  0  count  6  metric  [accuracy 0.55803573]
    Speedometer: epoch  0  count  7  metric  [accuracy 0.5625]
    Speedometer: epoch  0  count  8  metric  [accuracy 0.5798611]
    Speedometer: epoch  0  count  9  metric  [accuracy 0.584375]
    Speedometer: epoch  0  count  10  metric  [accuracy 0.57670456]
    Speedometer: epoch  0  count  11  metric  [accuracy 0.5807292]
    Speedometer: epoch  0  count  12  metric  [accuracy 0.5793269]
    Speedometer: epoch  1  count  1  metric  [accuracy 0.5625]
    Speedometer: epoch  1  count  2  metric  [accuracy 0.5520833]
    Speedometer: epoch  1  count  3  metric  [accuracy 0.5859375]
    Speedometer: epoch  1  count  4  metric  [accuracy 0.59375]
    Speedometer: epoch  1  count  5  metric  [accuracy 0.6145833]
    Speedometer: epoch  1  count  6  metric  [accuracy 0.625]
    Speedometer: epoch  1  count  7  metric  [accuracy 0.640625]
    Speedometer: epoch  1  count  8  metric  [accuracy 0.6527778]
    Speedometer: epoch  1  count  9  metric  [accuracy 0.653125]
    Speedometer: epoch  1  count  10  metric  [accuracy 0.6448864]
    Speedometer: epoch  1  count  11  metric  [accuracy 0.640625]
    Speedometer: epoch  1  count  12  metric  [accuracy 0.6418269]
    Speedometer: epoch  2  count  1  metric  [accuracy 0.671875]
    Speedometer: epoch  2  count  2  metric  [accuracy 0.7083333]
    Speedometer: epoch  2  count  3  metric  [accuracy 0.7109375]
    Speedometer: epoch  2  count  4  metric  [accuracy 0.725]
    Speedometer: epoch  2  count  5  metric  [accuracy 0.7239583]
    Speedometer: epoch  2  count  6  metric  [accuracy 0.71875]
    Speedometer: epoch  2  count  7  metric  [accuracy 0.734375]
    Speedometer: epoch  2  count  8  metric  [accuracy 0.7361111]
    Speedometer: epoch  2  count  9  metric  [accuracy 0.721875]
    Speedometer: epoch  2  count  10  metric  [accuracy 0.71022725]
    Speedometer: epoch  2  count  11  metric  [accuracy 0.6979167]
    Speedometer: epoch  2  count  12  metric  [accuracy 0.7019231]





    #object[org.apache.mxnet.module.Module 0x73c42ae5 "org.apache.mxnet.module.Module@73c42ae5"]


