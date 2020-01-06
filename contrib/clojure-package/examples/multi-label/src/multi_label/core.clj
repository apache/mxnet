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

(ns multi-label.core
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.context :as context])
  (:import (org.apache.mxnet DataIter)
           (java.util NoSuchElementException))
  (:gen-class))

(def data-dir "data/")
(def batch-size 100)
(def num-epoch 1)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "../../scripts/get_mnist_data.sh"))

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
(defn build-network []
  (let [fc3 (as-> (sym/variable "data") data
              (sym/fully-connected "fc1" {:data data :num-hidden 128})
              (sym/activation "relu1" {:data data :act-type "relu"})
              (sym/fully-connected "fc2" {:data data :num-hidden 64})
              (sym/activation "relu2" {:data data :act-type "relu"})
              (sym/fully-connected "fc3" {:data data :num-hidden 10}))
        sm1 (sym/softmax-output "softmax1" {:data fc3})
        sm2 (sym/softmax-output "softmax2" {:data fc3})]
    (sym/group [sm1 sm2])))

;;; provide an override proxy to the DataIter Scala class
(def multi-train-data (let [data-iter train-data]
                        (proxy [DataIter] []
                          (hasNext []
                            (mx-io/has-next? data-iter))
                          (next []
                            (if (mx-io/has-next? data-iter)
                              (let [batch (mx-io/next data-iter)]
                                (mx-io/data-batch {:data (util/scala-vector->vec
                                                          (.getData data-iter))
                                                   :label (let [label (first
                                                                       (util/scala-vector->vec (.getLabel data-iter)))]
                                                            [label label])
                                                   :index (util/scala-vector->vec
                                                           (.getIndex data-iter))
                                                   :pad (.pad batch)}))
                              (throw (new NoSuchElementException))))
                          (reset []
                            (mx-io/reset data-iter))
                          (batchSize []
                            (.batchSize data-iter))
                          (getData []
                            (.getData data-iter))
                          (getLabel []
                            (let [label (first  (util/scala-vector->vec (.getLabel data-iter)))]                              (util/vec->indexed-seq [label label])))
                          (getIndex []
                            (.getIndex data-iter))
                          (getPad []
                            (.getPad data-iter))
                          (provideLabel []
                            (let [shape (->> (mx-io/provide-label data-iter)
                                             (first)
                                             (vals)
                                             last)]
                              (util/list-map
                               {"softmax1_label" (mx-shape/->shape shape)
                                "softmax2_label" (mx-shape/->shape shape)})))
                          (provideData []
                            (.provideData data-iter)))))

(defn train [devs]
  (let [network (build-network)
        data-and-labels     (->> (into (mx-io/provide-data multi-train-data)
                                       (mx-io/provide-label multi-train-data))
                                 (mapcat vals)
                                 (apply hash-map))
        [arg-shapes output-shapes aux-shapes] (sym/infer-shape network data-and-labels)
        arg-names (sym/list-arguments network)
        aux-names (sym/list-auxiliary-states network)
        arg-params (zipmap arg-names (mapv #(ndarray/empty %) arg-shapes))
        aux-params (zipmap aux-names (mapv #(ndarray/empty %) aux-shapes))
        metric (eval-metric/custom-metric
                (fn [labels preds]
                  (println "Carin labels " labels)
                  (println "Carin preds " preds)
                  (float 0.5))
                "multi-accuracy")
        mod (-> (m/module network {:contexts devs})
                (m/bind {:data-shapes (mx-io/provide-data multi-train-data)
                         :label-shapes (mx-io/provide-label multi-train-data)})
                (m/init-params {:arg-params arg-params :aux-params aux-params})
                (m/init-optimizer))]
    (doseq [i (range 1)]
      (println "Doing epoch " i)
      (let [acc  (mx-io/reduce-batches
                  multi-train-data
                  (fn [r b]
                    (let [labels (mx-io/batch-label b)
                          preds (-> (m/forward mod b)
                                    (m/outputs))
                          accs (mapv (fn [p l]
                                       (let [pred-label (->> (ndarray/argmax-channel (first p))
                                                             (ndarray/->vec)
                                                             (mapv int))
                                             label (->> (ndarray/->vec l)
                                                        (mapv int))]
                                         (* 1.0 (apply + (mapv (fn [pl l] (if (= pl l) 1 0))
                                                               pred-label label)))))
                                     preds labels)]
                      (-> mod
                          (m/backward)
                          (m/update))
                      (-> r
                          (update :sum #(mapv (fn [o n] (+ o n)) % accs))
                          (update :batch-num inc))))
                  {:sum [0 0] :batch-num 0})]
        (println "Multi-accuracy " acc)
        (println "Multi-accuracy " (mapv #(/ % (:batch-num acc)) (:sum acc)))))))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Training...")
    (println "Running with context devices of" devs)
    (train devs)))

(comment
  (train [(context/cpu)]))
