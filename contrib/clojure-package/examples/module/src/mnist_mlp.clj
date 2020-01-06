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

(ns mnist-mlp
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:gen-class))

(def data-dir "data/")
(def batch-size 10)
(def num-epoch 5)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "../../scripts/get_mnist_data.sh"))
;; for save checkpoints load checkpoints
(io/make-parents "model/dummy.txt")

;;; Load the MNIST datasets
(defonce train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                       :label (str data-dir "train-labels-idx1-ubyte")
                                       :label-name "softmax_label"
                                       :input-shape [784]
                                       :batch-size batch-size
                                       :shuffle true
                                       :flat true
                                       :silent false
                                       :seed 10}))

(defonce test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                      :label (str data-dir "t10k-labels-idx1-ubyte")
                                      :input-shape [784]
                                      :batch-size batch-size
                                      :flat true
                                      :silent false}))
(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden 64})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/fully-connected "fc3" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))

(defn- print-header [message]
  (println "")
  (println "=================")
  (println (str "  " message))
  (println "=================")
  (println ""))

(defn run-intermediate-level-api [& {:keys [devs load-model-epoch]}]

  (let [header "Running Intermediate Level API"]
    (print-header (if load-model-epoch (str header " and loading from previous epoch " load-model-epoch)
                      header)))

  (let [save-prefix "model/mnist-mlp"
        mod (if load-model-epoch
              (do
                (println "Loading from checkpoint of epoch " load-model-epoch)
                (m/load-checkpoint {:contexts devs :prefix save-prefix :epoch load-model-epoch}))
              (m/module (get-symbol) {:contexts devs}))
        metric (eval-metric/accuracy)]
    (-> mod
        (m/bind {:data-shapes (mx-io/provide-data-desc train-data) :label-shapes (mx-io/provide-label-desc train-data)})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.01 :momentum 0.9})}))

    (doseq [epoch-num (range num-epoch)]
      (println "starting epoch " epoch-num)
      (mx-io/do-batches
       train-data
       (fn [batch]
         (-> mod
             (m/forward batch)
             (m/update-metric metric (mx-io/batch-label batch))
             (m/backward)
             (m/update))))
      (println "result for epoch " epoch-num " is " (eval-metric/get-and-reset metric))
      (m/save-checkpoint mod {:prefix save-prefix :epoch epoch-num :save-opt-states true}))))

(defn run-high-level-api [devs]
  (print-header "Running High Level API")

  (let [mod (m/module (get-symbol) {:contexts devs})]
    ;;; note only one function for training
    (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch})

    ;;high level predict (just a dummy call but it returns a vector of results
    (m/predict mod {:eval-data test-data})

    ;;;high level score (returs the eval values)
    (let [score (m/score mod {:eval-data test-data :eval-metric (eval-metric/accuracy)})]
      (println "High level predict score is " score))))

(defn run-predication-and-calc-accuracy-manually [devs]
  ;;; Gathers all the predictions at once with `predict-every-batch`
  ;;; then cycles thorugh the batches and manually calculates the accuracy stats

  (print-header "Running Predicting and Calcing the Accuracy Manually")

  (let [mod (m/module (get-symbol) {:contexts devs})]
    ;;; note only one function for training
    (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch})
    (let [preds (m/predict-every-batch mod {:eval-data test-data})
          stats (mx-io/reduce-batches test-data
                                      (fn [r b]
                                        (let [pred-label (->> (ndarray/argmax-channel (first (get preds (:index r))))
                                                              (ndarray/->vec)
                                                              (mapv int))
                                              label (->> (mx-io/batch-label b)
                                                         (first)
                                                         (ndarray/->vec)
                                                         (mapv int))
                                              acc-sum (apply + (mapv (fn [pl l] (if (= pl l) 1 0))
                                                                     pred-label label))]
                                          (-> r
                                              (update :index inc)
                                              (update :acc-cnt (fn [v] (+ v (count pred-label))))
                                              (update :acc-sum (fn [v] (+ v
                                                                          (apply + (mapv (fn [pl l] (if (= pl l) 1 0))
                                                                                         pred-label label))))))))
                                      {:acc-sum 0 :acc-cnt 0 :index 0})]
      (println "Stats: " stats)
      (println "Accuracy: " (/ (:acc-sum stats)
                               (* 1.0 (:acc-cnt stats)))))))

(defn run-prediction-iterator-api [devs]
  ;;Cycles through all the batchs and manually predicts and prints out the accuracy
  ;;using `predict-batch`

  (print-header "Running the Prediction Iterator API and Calcing the Accuracy Manually")

  (let [mod (m/module (get-symbol) {:contexts devs})]
    ;;; note only one function for training
    (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch})
    (mx-io/reduce-batches test-data
                          (fn [r b]
                            (let [preds (m/predict-batch mod b)
                                  pred-label (->> (ndarray/argmax-channel (first preds))
                                                  (ndarray/->vec)
                                                  (mapv int))
                                  label (->> (mx-io/batch-label b)
                                             (first)
                                             (ndarray/->vec)
                                             (mapv int))
                                  acc (/ (apply + (mapv (fn [pl l] (if (= pl l) 1 0)) pred-label label))
                                         (* 1.0 (count pred-label)))]
                              (println "Batch " r " acc: " acc)
                              (inc r))))))

(defn run-all [devs]
  (run-intermediate-level-api :devs devs)
  (run-intermediate-level-api :devs devs :load-model-epoch (dec num-epoch))
  (run-high-level-api devs)
  (run-prediction-iterator-api devs)
  (run-predication-and-calc-accuracy-manually devs))

(defn -main
  [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Running Module MNIST example")
    (println "Running with context devices of" devs)
    (run-all devs)))

(comment

  ;;; run all the example functions
  (run-all [(context/cpu)])

  ;;; run for the number of epochs
  (run-intermediate-level-api :devs [(context/cpu)])
  ;;=> starting epoch  0
  ;;=> result for epoch  0  is  [accuracy 0.8531333]
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved checkpoint to model/mnist-mlp-0000.params
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved optimizer state to model/mnist-mlp-0000.states
  ;;=> ....
  ;;=> starting epoch  4
  ;;=> result for epoch  4  is  [accuracy 0.91875]
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved checkpoint to model/mnist-mlp-0004.params
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved optimizer state to model/mnist-mlp-0004.states


  ;; load from the last saved file and run again
  (run-intermediate-level-api :devs [(context/cpu)] :load-model-epoch (dec num-epoch))
  ;;=> Loading from checkpoint of epoch  4
  ;;=> starting epoch  0
  ;;=> result for epoch  0  is  [accuracy 0.96258336]
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved checkpoint to model/mnist-mlp-0000.params
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved optimizer state to model/mnist-mlp-0000.states
  ;;=> ...
  ;;=> starting epoch  4
  ;;=> result for epoch  4  is  [accuracy 0.9819833]
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved checkpoint to model/mnist-mlp-0004.params
  ;;=> INFO  ml.dmlc.mxnet.module.Module: Saved optimizer state to model/mnist-mlp-0004.states

  (run-high-level-api [(context/cpu)])
  ;;=> ["accuracy" 0.9454]

  (run-prediction-iterator-api [(context/cpu)])
  ;;=> Batch  0  acc:  1.0
  ;;=> Batch  1  acc:  0.9
  ;;=> Batch  2  acc:  1.0
  ;;=> ...
  ;;=> Batch  999  acc:  1.0

  (run-predication-and-calc-accuracy-manually [(context/cpu)])
  ;;=> Stats:  {:acc-sum 9494, :acc-cnt 10000, :index 1000}
  ;;=> Accuracy:  0.9494
)

