;;
;; Licensed to the Apache Software Foundation (ASF) under one or more
;; contributor license agreements.  See the NOTICE file distributed with
;; this work for additional information regarding copyright ownership.
;; The ASF licenses this file to You under the Apache License, Version 2.0
;; (the "License"); you may not use this file except in compliance with
;; the License.  You may obtain a copy of the License at
;;
;;    http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.
;;

(ns bert.bert-sentence-classification
  (:require [bert.util :as bert-util]
            [clojure-csv.core :as csv]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]))

(def model-path-prefix "data/static_bert_base_net")
;; epoch number of the model
;; the maximum length of the sequence
(def seq-length 128)

(defn pre-processing
  "Preprocesses the sentences in the format that BERT is expecting"
  [idx->token token->idx train-item]
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

(defn fine-tune-model
  "msymbol: the pretrained network symbol
   num-classes: the number of classes for the fine-tune datasets
   dropout: The dropout rate amount"
  [msymbol {:keys [num-classes dropout]}]
  (as-> msymbol data
    (sym/dropout {:data data :p dropout})
    (sym/fully-connected "fc-finetune" {:data data :num-hidden num-classes})
    (sym/softmax-output "softmax" {:data data})))

(defn slice-inputs-data
  "Each sentence pair had to be processed as a row. This breaks all
  the rows up into a column for creating a NDArray"
  [processed-datas n]
  (->> processed-datas
       (mapv #(nth (:input-batch %) n))
       (flatten)
       (into [])))

(defn get-raw-data []
  (csv/parse-csv (string/replace (slurp "data/dev.tsv") "\"" "")
               :delimiter \tab
               :strict true))

(defn prepare-data
  "This prepares the senetence pairs into NDArrays for use in NDArrayIterator"
  []
  (let [raw-file (get-raw-data)
        vocab (bert-util/get-vocab)
        idx->token (:idx->token vocab)
        token->idx (:token->idx vocab)
        data-train-raw (->> raw-file
                            (mapv #(vals (select-keys % [3 4 0])))
                            (rest) ;;drop header
                            (into []))
        processed-datas (mapv #(pre-processing idx->token token->idx %) data-train-raw)]
    {:data0s (slice-inputs-data processed-datas 0)
     :data1s (slice-inputs-data processed-datas 1)
     :data2s (slice-inputs-data processed-datas 2)
     :labels (->> (mapv :label processed-datas)
                  (flatten)
                  (into []))
     :train-num (count processed-datas)}))

(defn train
  "Trains (fine tunes) the sentence pairs for a classification task on the BERT Base model"
  [dev num-epoch]
  (let [bert-base (m/load-checkpoint {:prefix model-path-prefix :epoch 0})
        model-sym (fine-tune-model (m/symbol bert-base) {:num-classes 2 :dropout 0.1})
        {:keys [data0s data1s data2s labels train-num]} (prepare-data)
        batch-size 32
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
                                     :layout layout/N})
        train-data  (mx-io/ndarray-iter {data-desc0 (ndarray/array data0s [train-num seq-length]
                                                                   {:ctx dev})
                                         data-desc1 (ndarray/array data1s [train-num seq-length]
                                                                   {:ctx dev})
                                         data-desc2 (ndarray/array data2s [train-num]
                                                                   {:ctx dev})}
                                        {:label {label-desc (ndarray/array labels [train-num]
                                                                           {:ctx dev})}
                                         :data-batch-size batch-size})
        model (m/module model-sym {:contexts [dev]
                                   :data-names ["data0" "data1" "data2"]})]
    (m/fit model {:train-data train-data  :num-epoch num-epoch
                  :fit-params (m/fit-params {:allow-missing true
                                             :arg-params (m/arg-params bert-base)
                                             :aux-params (m/aux-params bert-base)
                                             :optimizer (optimizer/adam {:learning-rate 5e-6 :episilon 1e-9})
                                             :batch-end-callback (callback/speedometer batch-size 1)})})))

(defn -main [& args]
  (let [[dev-arg num-epoch-arg] args
        dev (if (= dev-arg ":gpu") (context/gpu) (context/cpu))
        num-epoch (if num-epoch-arg (Integer/parseInt num-epoch-arg) 3)]
    (println "Running example with " dev " and " num-epoch " epochs ")
    (train dev num-epoch)))

(comment

  (train (context/cpu 0) 3)
  (m/save-checkpoint model {:prefix "fine-tune-sentence-bert" :epoch 3}))
