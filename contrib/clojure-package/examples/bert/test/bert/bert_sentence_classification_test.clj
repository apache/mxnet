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


(ns bert.bert-sentence-classification-test
  (:require [bert.bert-sentence-classification :refer :all]
            [clojure-csv.core :as csv]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.module :as m]))

(def model-dir "data/")

(when-not (.exists (io/file (str model-dir "static_bert_qa-0002.params")))
  (println "Downloading bert qa data")
  (sh "./get_bert_data.sh"))

(defn get-slim-raw-data []
  (take 32 (csv/parse-csv (slurp "data/dev.tsv") :delimiter \tab)))

(deftest train-test
  (with-redefs [get-raw-data get-slim-raw-data]
    (let [dev (context/default-context)
          num-epoch 1
          bert-base (m/load-checkpoint {:prefix model-path-prefix :epoch 0})
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
                                               :batch-end-callback (callback/speedometer batch-size 1)})})
      (is (< 0.5 (-> (m/score model {:eval-data train-data :eval-metric (eval-metric/accuracy) })
                     (last)))))))
