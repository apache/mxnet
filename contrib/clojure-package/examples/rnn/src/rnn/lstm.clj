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

(ns rnn.lstm
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]))

(defn lstm-param [i2h-weight i2h-bias
                  h2h-weight h2h-bias]
  {:i2h-weight i2h-weight :i2h-bias i2h-bias
   :h2h-weight h2h-weight :h2h-bias h2h-bias})

(defn lstm-state [c h]
  {:c c :h h})

(defn lstm [num-hidden in-data prev-state param seq-idx layer-idx dropout]
  (let [in-dataa (if (pos? dropout)
                   (sym/dropout {:data in-data :p dropout})
                   in-data)
        i2h (sym/fully-connected (str "t" seq-idx "_l" layer-idx "_i2h")
                                 {:data in-dataa :weight (:i2h-weight param)
                                  :bias (:i2h-bias param) :num-hidden (* num-hidden 4)})
        h2h (sym/fully-connected (str "t" seq-idx "_l" layer-idx "_h2h")
                                 {:data (:h prev-state) :weight (:h2h-weight param)
                                  :bias (:h2h-bias param) :num-hidden (* num-hidden 4)})
        gates (sym/+ i2h h2h)
        slice-gates (sym/slice-channel (str "t" seq-idx "_l" layer-idx "_slice")
                                       {:data gates :num-outputs 4})
        in-gate (sym/activation {:data (sym/get slice-gates 0) :act-type "sigmoid"})
        in-transform (sym/activation {:data (sym/get slice-gates 1) :act-type "tanh"})
        forget-gate (sym/activation {:data (sym/get slice-gates 2) :act-type "sigmoid"})
        out-gate (sym/activation {:data (sym/get slice-gates 3) :act-type "sigmoid"})
        next-c (sym/+ (sym/* forget-gate (:c prev-state))
                      (sym/* in-gate in-transform))
        next-h (sym/* out-gate (sym/activation {:data next-c :act-type "tanh"}))]
    (lstm-state next-c next-h)))

(defn lstm-unroll [num-lstm-layer seq-len input-size num-hidden num-embed num-label dropout]
  (let [embed-weight (sym/variable "embed_weight")
        cls-weight (sym/variable "cls_weight")
        cls-bias (sym/variable "cls_bias")
        param-cells (mapv (fn [i]
                            (lstm-param (sym/variable (str "l" i "_i2h_weight"))
                                        (sym/variable (str "l" i "_i2h_bias"))
                                        (sym/variable (str "l" i "_h2h_weight"))
                                        (sym/variable (str "l" i "_h2h_bias"))))
                          (range 0 num-lstm-layer))
        last-states (mapv (fn [i]
                            (lstm-state (sym/variable (str "l" i "_init_c_beta"))
                                        (sym/variable (str "l" i "_init_h_beta"))))
                          (range 0 num-lstm-layer))
        ;; embedding layer
        data (sym/variable "data")
        label (sym/variable "softmax_label")
        embed (sym/embedding "embed" {:data data :input-dim input-size :weight embed-weight
                                      :output-dim num-embed})
        wordvec (sym/slice-channel {:data embed :num-outputs seq-len :squeeze-axis 1})
        dp-ratio 0
        ;; stack lstm
        hidden-all (doall (for [seq-idx (range seq-len)]
                            (let [hidden (:h (last (loop [i 0
                                                          hidden (sym/get wordvec seq-idx)
                                                          next-states []]
                                                     (if (= i num-lstm-layer)
                                                       next-states
                                                       (let [dp-ratio (if (zero? i) 0 dropout)
                                                             next-state (lstm num-hidden
                                                                              hidden
                                                                              (get last-states i)
                                                                              (get param-cells i)
                                                                              seq-idx
                                                                              i
                                                                              dp-ratio)]
                                                         (recur (inc i)
                                                                (:h next-state)
                                                                (conj next-states next-state)))))))]
                              (if (pos? dropout)
                                (sym/dropout {:data hidden :p dropout})
                                hidden))))
        hidden-concat (sym/concat "concat" nil hidden-all {:dim 0})
        pred (sym/fully-connected "pred" {:data hidden-concat :num-hidden num-label
                                          :weight cls-weight :bias cls-bias})
        label (sym/transpose {:data label})
        label (sym/reshape {:data label :target-shape [0]})
        sm (sym/softmax-output "softmax" {:data pred :label label})]
    sm))

(defn lstm-inference-symbol [num-lstm-layer input-size num-hidden
                             num-embed num-label dropout]
  (let [seq-idx 0
        embed-weight (sym/variable "embed_weight")
        cls-weight (sym/variable "cls_weight")
        cls-bias (sym/variable "cls_bias")
        param-cells (mapv (fn [i]
                            (lstm-param (sym/variable (str "l" i "_i2h_weight"))
                                        (sym/variable (str "l" i "_i2h_bias"))
                                        (sym/variable (str "l" i "_h2h_weight"))
                                        (sym/variable (str "l" i "_h2h_bias"))))
                          (range 0 num-lstm-layer))
        last-states (mapv (fn [i]
                            (lstm-state (sym/variable (str "l" i "_init_c_beta"))
                                        (sym/variable (str "l" i "_init_h_beta"))))
                          (range 0 num-lstm-layer))
        data (sym/variable "data")
        dp-ratio 0
        ;; stack lstm
        next-states (loop [i 0
                           hidden (sym/embedding "embed" {:data data :input-dim input-size :weight embed-weight :output-dim num-embed})
                           next-states []]
                      (if (= i num-lstm-layer)
                        next-states
                        (let [dp-ratio (if (zero? i) 0 dropout)
                              next-state (lstm num-hidden
                                               hidden
                                               (get last-states i)
                                               (get param-cells i)
                                               seq-idx
                                               i
                                               dp-ratio)]
                          (recur (inc i)
                                 (:h next-state)
                                 (conj next-states next-state)))))
        ;;; decoder
        hidden (:h (last next-states))
        hidden (if (pos? dropout) (sym/dropout {:data hidden :p dropout}) hidden)
        fc (sym/fully-connected "pred" {:data hidden :num-hidden num-label
                                        :weight cls-weight :bias cls-bias})
        sm (sym/softmax-output "softmax" {:data fc})
        outs (into [sm] (mapcat (fn [next-s] (vals next-s)) next-states))]
    (sym/group outs)))

(defn lstm-inference-model [{:keys [num-lstm-layer input-size num-hidden
                                    num-embed num-label arg-params
                                    ctx dropout]
                             :or {ctx (context/cpu)
                                  dropout 0.0}}]

  (let [lstm-sym (lstm-inference-symbol num-lstm-layer
                                        input-size
                                        num-hidden
                                        num-embed
                                        num-label
                                        dropout)
        batch-size 1
        init-c (into {} (map (fn [l]
                               {(str "l" l "_init_c_beta") [batch-size num-hidden]})
                             (range num-lstm-layer)))
        init-h (into {} (map (fn [l]
                               {(str "l" l "_init_h_beta") [batch-size num-hidden]}))
                     (range num-lstm-layer))
        data-shape {"data" [batch-size]}
        input-shape (merge init-c init-h data-shape)
        exec (sym/simple-bind lstm-sym ctx input-shape)
        exec-arg-map (executor/arg-map exec)
        states-map (zipmap (mapcat (fn [i] [(str "l" i "_init_c_beta")
                                            (str "l" i "_init_h_beta")])
                                   (range num-lstm-layer))
                           (rest (executor/outputs exec)))]
    (doseq [[k v] arg-params]
      (if-let [target-v (get exec-arg-map k)]
        (when (and (not (get input-shape k))
                   (not= "softmax_label" k))
          (ndarray/copy-to v target-v))))
    {:exec exec
     :states-map states-map}))

(defn forward [{:keys [exec states-map] :as lstm-model} input-data new-seq]
  (when new-seq
    (doseq [[k v] states-map]
      (ndarray/set (get (executor/arg-map exec) k) 0)))
  (do
    (ndarray/copy-to input-data (get (executor/arg-map exec) "data"))
    (executor/forward exec)
    (doseq [[k v] states-map]
      (ndarray/copy-to v (get (executor/arg-map exec) k)))
    (first (executor/outputs exec))))
