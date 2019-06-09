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

(ns cnn-text-classification.classifier
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [cnn-text-classification.data-helper :as data-helper]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context])
  (:gen-class))

(def data-dir "data/")
(def mr-dataset-path "data/mr-data") ;; the MR polarity dataset path
(def glove-file-path "data/glove/glove.6B.50d.txt")
(def num-filter 100)
(def num-label 2)
(def dropout 0.5)



(when-not (.exists (io/file (str data-dir)))
  (do (println "Retrieving data for cnn text classification...") (sh "./get_data.sh")))

(defn shuffle-data [test-num {:keys [data label sentence-count sentence-size embedding-size]}]
  (println "Shuffling the data and splitting into training and test sets")
  (println {:sentence-count sentence-count
            :sentence-size sentence-size
            :embedding-size embedding-size})
  (let [shuffled (shuffle (map #(vector %1 %2) data label))
        train-num (- (count shuffled) test-num)
        training (into [] (take train-num shuffled))
        test (into [] (drop train-num shuffled))]
    {:training {:data  (ndarray/array (into [] (flatten (mapv first training)))
                                      [train-num 1 sentence-size embedding-size]) ;; has to be channel x y
                :label (ndarray/array (into [] (flatten (mapv last  training)))
                                      [train-num])}
     :test {:data  (ndarray/array (into [] (flatten (mapv first test)))
                                  [test-num 1 sentence-size embedding-size]) ;; has to be channel x y
            :label (ndarray/array (into [] (flatten (mapv last  test)))
                                  [test-num])}}))

(defn make-filter-layers [{:keys [input-x num-embed sentence-size] :as config}
                          filter-size]
  (as-> (sym/convolution {:data input-x
                          :kernel [filter-size num-embed]
                          :num-filter num-filter}) data
    (sym/activation {:data data :act-type "relu"})
    (sym/pooling {:data data
                  :pool-type "max"
                  :kernel [(inc (- sentence-size filter-size)) 1]
                  :stride [1 1]})))

;;; convnet with multiple filter sizes
;; from Convolutional Neural Networks for Sentence Classification by Yoon Kim
(defn get-multi-filter-convnet [num-embed sentence-size batch-size]
  (let [filter-list [3 4 5]
        input-x (sym/variable "data")
        polled-outputs (mapv #(make-filter-layers {:input-x input-x :num-embed num-embed :sentence-size sentence-size} %) filter-list)
        total-filters (* num-filter (count filter-list))
        concat (sym/concat "concat" nil polled-outputs {:dim 1})
        hpool (sym/reshape "hpool" {:data concat :target-shape [batch-size total-filters]})
        hdrop (if (pos? dropout) (sym/dropout "hdrop" {:data hpool :p dropout}) hpool)
        fc (sym/fully-connected  "fc1" {:data hdrop :num-hidden num-label})]
    (sym/softmax-output "softmax" {:data fc})))

(defn train-convnet [{:keys [devs embedding-size batch-size test-size num-epoch max-examples]}]
  (let [glove (data-helper/load-glove glove-file-path) ;; you can also use word2vec
        ms-dataset (data-helper/load-ms-with-embeddings mr-dataset-path embedding-size glove max-examples)
        sentence-size (:sentence-size ms-dataset)
        shuffled (shuffle-data test-size ms-dataset)
        train-data (mx-io/ndarray-iter [(get-in shuffled [:training :data])]
                                       {:label [(get-in shuffled [:training :label])]
                                        :label-name "softmax_label"
                                        :data-batch-size batch-size
                                        :last-batch-handle "pad"})
        test-data (mx-io/ndarray-iter [(get-in shuffled [:test :data])]
                                      {:label [(get-in  shuffled [:test :label])]
                                       :label-name "softmax_label"
                                       :data-batch-size batch-size
                                       :last-batch-handle "pad"})]
    (let [mod (m/module (get-multi-filter-convnet embedding-size sentence-size batch-size) {:contexts devs})]
      (println "Getting ready to train for " num-epoch " epochs")
      (println "===========")
      (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch
                  :fit-params (m/fit-params {:optimizer (optimizer/adam)})}))))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
  ;;; omit max-examples if you want to run all the examples in the movie review dataset
    ;; to limit mem consumption set to something like 1000 and adjust test size to 100
    (println "Running with context devices of" devs)
    (train-convnet {:devs devs :embedding-size 50 :batch-size 10 :test-size 100 :num-epoch 10 :max-examples 1000})
    ;; runs all the examples
    #_(train-convnet {:embedding-size 50 :batch-size 100 :test-size 1000 :num-epoch 10})))

(comment
  (train-convnet {:devs devs :embedding-size 50 :batch-size 10 :test-size 100 :num-epoch 10 :max-examples 1000}))

