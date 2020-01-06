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

(ns rnn.train-char-rnn
  (:require  [clojure.string :as string]
             [clojure.java.shell :refer [sh]]
             [rnn.util :as util]
             [rnn.lstm :as lstm]
             [rnn.test-char-rnn :as test-rnn]
             [org.apache.clojure-mxnet.context :as context]
             [org.apache.clojure-mxnet.callback :as callback]
             [org.apache.clojure-mxnet.executor :as executor]
             [org.apache.clojure-mxnet.eval-metric :as eval-metric]
             [org.apache.clojure-mxnet.io :as mx-io]
             [org.apache.clojure-mxnet.initializer :as init]
             [org.apache.clojure-mxnet.ndarray :as ndarray]
             [org.apache.clojure-mxnet.optimizer :as optimizer]
             [org.apache.clojure-mxnet.symbol :as sym]
             [org.apache.clojure-mxnet.module :as m])
  (:gen-class))

;;https://github.com/apache/incubator-mxnet/blob/master/example/rnn/old/char-rnn.ipynb

(when-not (.exists (clojure.java.io/file "data"))
  (do (println "Retrieving data...") (sh "./get_data.sh")))

;; batch size for training
(def batch-size 32)
;; we can support various length input
;; for this problem, we cut each input sentence to length of 129
;; so we only need a fixed lenght bucket
(def buckets [129])
;;hidden unit in LSTM cell
(def num-hidden 512)
;; embedding dim which is map a char to a 256 dim vector
(def num-embed 256)
;; number of lstm layer
(def num-lstm-layer 3)
;; we will show a quick demo in 2 epoch and we will see the result
;; by training 75 epoch
(def num-epoch 75)
;; learning rate
(def learning-rate 0.01)
;; we will use pure sgd without momentum
(def momentum 0.0)

(def ctx (context/cpu)) ;; change to gpu if desired
(def data-path "data/obama.txt")
(def vocab (util/build-vocab data-path))

;; generate the symbol for a length
(defn sym-gen [seq-len]
  (lstm/lstm-unroll num-lstm-layer seq-len (inc (count vocab))
                    num-hidden num-embed (inc (count vocab)) 0.2))

;;; in the case of this fixed bucketing that only uses one bucket size - it is the equivalent of padpadding all sentences to a fixed length.
;; we are going to use ndarray-iter for this
;; converting the bucketing-iter over to use is todo. We could either push for the example Scala one to be included in the base package and interop with that (which would be nice for other rnn needs too) or hand convert it over ourselves


(defn build-training-data [path]
  (let [content (slurp path)
        sentences (string/split content #"\n")
        max-length (first buckets)
        padding-int 0]
    (doall (for [sentence sentences]
             (let [ids (mapv #(get vocab %) sentence)]
               (if (>= (count ids) max-length)
                 (into [] (take max-length ids))
                 (into ids (repeat (- max-length (count ids)) 0))))))))

(defn build-labels [train-data]
    ;; want to learn the next char some rotate by 1
  (doall (mapv (fn [sent-data] (conj (into [] (rest sent-data)) 0))
               train-data)))

(defn data-desc->map [data-desc]
  (->>  data-desc
        (map vals)
        (first)
        (apply hash-map)))

(defn train [devs]
  (let [;; initialize the states for the lstm
        init-c (into {} (map (fn [l]
                               {(str "l" l "_init_c_beta") [batch-size num-hidden]})
                             (range num-lstm-layer)))
        init-h (into {} (map (fn [l]
                               {(str "l" l "_init_h_beta") [batch-size num-hidden]}))
                     (range num-lstm-layer))
        init-states (merge init-c init-h)
        train-data (build-training-data data-path)
        labels (build-labels train-data)
        sent-len (first buckets)
        train-iter (mx-io/ndarray-iter [(ndarray/array (flatten train-data)
                                                       [(count train-data) sent-len])]
                                       {:label [(ndarray/array (flatten labels)
                                                               [(count labels) sent-len])]
                                        :label-name "softmax_label"
                                        :data-batch-size batch-size
                                        :last-batch-handle "pad"})
        data-and-labels (merge (data-desc->map (mx-io/provide-data-desc train-iter))
                               (data-desc->map (mx-io/provide-label-desc train-iter))
                               init-states)
        init-states-data (mapv (fn [[k v]] (ndarray/zeros v {:ctx ctx})) init-states)
        rnn-sym (sym-gen (first buckets))

        rnn-mod (-> (m/module rnn-sym {:contexts devs})
                    (m/bind {:data-shapes (into (mx-io/provide-data-desc train-iter)
                                                (mapv (fn [[k v]] {:name k :shape v}) init-states))
                             :label-shapes (mx-io/provide-label-desc train-iter)})
                    (m/init-params {:initializer (init/xavier {:factor-type "in" :magnitude 2.34})})
                    (m/init-optimizer {:optimizer (optimizer/adam {:learning-rate learning-rate :wd 0.0001})}))
        metric (eval-metric/custom-metric
                (fn [label pred]
                  (let [labels (ndarray/->vec (ndarray/transpose label))
                        pred-shape (ndarray/shape-vec pred)
                        size (apply * (ndarray/shape-vec label))
                        preds (mapv #(into [] %) (doall
                                                  (partition (last pred-shape) (ndarray/->vec pred))))
                        results (map-indexed
                                 (fn [i l]
                                   (get-in preds [i (int l)]))
                                 labels)
                        result (->> results
                                    (mapv #(Math/max (float 1e-10) (float %)))
                                    (mapv #(Math/log %))
                                    (mapv #(* -1.0 %))
                                    (apply +))]
                    (float (Math/exp (/ result (count labels))))))

                "perplexity")]

    ;; Train for 1 epochs and then show the results of 75
    (doseq [epoch-num (range 1)]
      (println "Doing epoch " epoch-num)
      (mx-io/reduce-batches
       train-iter
       (fn [batch-num batch]
         (let [batch (mx-io/next train-iter)]
           (-> rnn-mod
               (m/forward (mx-io/data-batch {:data (into (mx-io/batch-data batch) init-states-data)
                                             :label (mx-io/batch-label batch)}))
               (m/update-metric metric (mx-io/batch-label batch))
               (m/backward)
               (m/update))
           (when (zero? (mod batch-num 10))
             (println "Eval metric for batch-num " batch-num " is " (eval-metric/get metric)))
           (inc batch-num))))
      (println "Finished epoch " epoch-num)
      #_(println "Eval-metric " (eval-metric/get-and-reset metric))
      (m/save-checkpoint rnn-mod {:prefix "train-obama" :epoch epoch-num})
      (println "Testing with random 200 chars ")
      (println "=====")
      (println  (test-rnn/rnn-test "train-obama" epoch-num 200 true))
      (println "====="))

    (println "Showing the result after 75 epochs (pre-trained)")
    (println (test-rnn/rnn-test "data/obama" 75 200 true))
    (println "=====")))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (train devs)))
