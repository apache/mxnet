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

(ns org.apache.clojure-mxnet.eval-metric
  (:refer-clojure :exclude [get update])
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet Accuracy TopKAccuracy F1 Perplexity MAE MSE RMSE CustomMetric CompositeEvalMetric)))

(defn accuracy
  "Basic Accuracy Metric"
  []
  (new Accuracy))

(defn top-k-accuracy
  "Calculate to k predications accuracy
  - top-k number of predicts (int)"
  [top-k]
  (new TopKAccuracy (int top-k)))

(defn f1
  "Calculate the F1 score of a binary classification problem."
  []
  (new F1))

(defn perplexity
  "Calculate perplexity
   - opts
    :ignore-label Index of invalid label to ignore when counting. Usually should be -1. Include
     all entries if None.
    :axis The axis from prediction that was used to
    compute softmax. Default is -1 which means use the last axis."
  ([{:keys [ignore-label axis] :as opts
     :or {axis -1}}]
   (new Perplexity
        (if ignore-label (util/->option (int ignore-label)) (util/->option nil))
        (int axis)))
  ([]
   (perplexity {})))

(defn mae
  "Calculate Mean Absolute Error loss"
  []
  (new MAE))

(defn mse
  "Calculate Mean Squared Error loss"
  []
  (new MSE))

(defn rmse
  "Calculate Root Mean Squred Error loss"
  []
  (new RMSE))

(defmacro custom-metric
  "Custom evaluation metric that takes a NDArray function.
   - f-eval Customized evaluation function that takes two ndarrays and returns a number
     function must be in the form of (fn [] ) clojure style
   - mname The name of the metric"
  [f-eval mname]
  `(new CustomMetric (util/scala-fn ~f-eval) ~mname))

(defn comp-metric
  "Create a metric instance composed out of several metrics"
  [metrics]
  (let [cm (CompositeEvalMetric.)]
    (doseq [m metrics] (.add cm m))
    cm))

(defn get
  "Get the values of the metric in as a map of {name value} pairs"
  [metric]
  (let [m (apply zipmap (-> (.get metric)
                            util/tuple->vec))]
    (if-not (instance? CompositeEvalMetric metric)
      (first m)
      m)))

(defn reset
  "clear the internal statistics to an initial state"
  [metric]
  (doto metric
    (.reset)))

(defn update
  "Update the internal evaluation"
  [metric labels preds]
  (doto metric
    (.update (util/vec->indexed-seq labels) (util/vec->indexed-seq preds))))

(defn get-and-reset
  "Get the values and then reset the metric"
  [metric]
  (let [v (get metric)]
    (reset metric)
    v))
