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

(ns org.apache.clojure-mxnet.module-test
  (:require [clojure.java.io :as io]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.monitor :as monitor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s]
            [clojure.test :refer :all]
            [clojure.reflect :as r]
            [clojure.string :as string]))

(deftest test-model-dtype
  (let [dtype dtype/FLOAT32
        dshape [3 8 7]
        s (sym/variable "data")
        s (sym/activation "act" {"__layout__" "TNC"} {:data s :act_type "relu"})

        mod (m/module s ["data"] nil [(context/cpu 0) (context/cpu 1)])]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape dshape :dtype dtype :layout "TNC"}]})
        (m/init-params)
        (m/forward {:data [(ndarray/ones dshape {:dtype dtype})]})
        (m/backward [(ndarray/ones dshape {:dtype dtype})]))
    (let [outputs  (-> mod (m/outputs) flatten)]
      (is (every? #(= dtype/FLOAT32 (ndarray/dtype %)) outputs)))))

(deftest test-module-input-grads
  (let [a (sym/variable "a" {:kwargs {"__layout__" "NC"}})
        b (sym/variable "b" {:kwargs {"__layout__" "NC"}})
        c (sym/variable "c" {:kwargs {"__layout__" "NC"}})
        c (sym/+ a (sym/+ (sym/* b 2) (sym/* c 3)))
        mod (m/module c ["b" "c" "a"] nil [(context/cpu 0) (context/cpu 1)])]
    (-> mod
        (m/bind {:data-shapes [{:name "b" :shape [5 5] :layout layout/NT}
                               {:name "c" :shape [5 5] :layout layout/NT}
                               {:name "a" :shape [5 5] :layout layout/NT}]
                 :inputs-need-grad true})
        (m/init-params)
        (m/forward {:data [(ndarray/ones [5 5])
                           (ndarray/ones [5 5])
                           (ndarray/ones [5 5])]
                    :label nil
                    :index nil
                    :pad 0})
        (m/backward [(ndarray/ones [5 5])]))
    (let [[a-grad b-grad c-grad] (m/input-grads-merged mod)]
      (is (every? #(= 1.0 %) (ndarray/->vec a-grad)))
      (is (every? #(= 2.0 %) (ndarray/->vec b-grad)))
      (is (every? #(= 3.0 %) (ndarray/->vec c-grad))))))

(deftest test-module-layout
  (let [s (sym/variable "data")
        s (sym/activation "act " {"__layout__" "TNC"} {:data s :act_type "relu"})
        dshape [3 8 7]
        mod (m/module s ["data"] nil [(context/cpu 0) (context/cpu 1)])]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape dshape :dtype dtype/FLOAT32 :layout "TNC"}]})
        (m/init-params)
        (m/forward {:data [(ndarray/ones dshape)]
                    :label nil
                    :index nil
                    :pad 0})
        (m/backward [(ndarray/ones dshape)]))
    (let [outputs-merged (m/outputs-merged mod)
          outputs (m/outputs mod)
          hd-shape [3 4 7]]
      (is (= dshape (-> outputs-merged first (ndarray/shape) (ndarray/->vec))))
      (is (every? #(= hd-shape (-> % ndarray/shape ndarray/->vec)) (flatten outputs))))))

(deftest test-module-save-load-single-device
  (let [s (sym/variable "data")
        s (sym/fully-connected {:data s :num-hidden 100})
        ;; single device
        mod (m/module s {:data-names ["data"] :label-names nil})]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1 :momentum 0.9})})
        (m/update)
        (m/save-checkpoint {:prefix "test" :epoch 0 :save-opt-states true}))
    (let [mod2 (m/load-checkpoint {:prefix "test" :epoch 0 :load-optimizer-states true})]
      (-> mod2
          (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
          (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1 :momentum 0.9})}))
      (is (= (-> mod m/symbol sym/to-json) (-> mod2 m/symbol sym/to-json)))
      (is (= (-> mod m/params first) (-> mod2 m/params first))))
    ;; arity 2 version of above. `load-optimizer-states` is `false` here by default,
    ;; but optimizers states aren't checked here so it's not relevant to the test outcome.
    (let [mod3 (m/load-checkpoint "test" 0)]
      (-> mod3
          (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
          (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1 :momentum 0.9})}))
      (is (= (-> mod m/symbol sym/to-json) (-> mod3 m/symbol sym/to-json)))
      (is (= (-> mod m/params first) (-> mod3 m/params first))))))

(deftest test-module-save-load-multi-device
  (let [s (sym/variable "data")
        s (sym/fully-connected {:data s :num-hidden 100})
        ;; multi device
        mod (m/module s {:data-names ["data"] :label-names nil
                         :contexts [(context/cpu 0) (context/cpu 1)]})]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1 :momentum 0.9})})
        (m/update)
        (m/save-checkpoint {:prefix "test" :epoch 0 :save-opt-states true}))

    (let [mod2 (m/load-checkpoint {:prefix "test" :epoch 0 :load-optimizer-states true})]
      (-> mod2
          (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
          (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1 :momentum 0.9})}))
      (is (= (-> mod m/symbol sym/to-json)  (-> mod2 m/symbol sym/to-json)))
      (is (= (-> mod m/params first) (-> mod2 m/params first))))))

(deftest test-module-reshape
  (let [s (sym/variable "data")
        s (sym/fully-connected "fc" {:data s :num-hidden 20})
        dshape [7 20]
        mod (m/module s ["data"] nil [(context/cpu 0) (context/cpu 1)])]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape dshape :layout "NT"}]})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 1.0})})
        (m/forward {:data [(ndarray/ones dshape)] :label nil :index nil :pad 0})
        (m/backward [(ndarray/ones dshape)])
        (m/update))
    (is (= dshape (-> (m/outputs-merged mod) first ndarray/shape mx-shape/->vec)))
    (is (every? #(= -1.0 %) (-> (m/params mod) (first) (get "fc_bias") (ndarray/->vec))))

    (let [dshape [14 20]]
      (-> mod
          (m/reshape [{:name "data" :shape dshape :layout "NT"}])
          (m/forward {:data [(ndarray/ones dshape)] :label nil :index nil :pad 0})
          (m/backward [(ndarray/ones dshape)])
          (m/update))
      (is (= dshape (-> (m/outputs-merged mod) first ndarray/shape mx-shape/->vec)))
      (is (every? #(< 1e-3 (- 3 %)) (-> mod m/params first (get "fc_bias") (ndarray/->vec)))))))

(deftest test-set-params
  (let [data (ndarray/array [0.05 0.1] [1 1 1 2])
        label (ndarray/array [0.01 0.99] [1 1 1 2])
        train-data (mx-io/ndarray-iter [data] {:label [label] :label-name "softmax_label"})
        x (as-> (sym/variable "data") v
            (sym/fully-connected "fc_0" {:data v :num-hidden 2})
            (sym/activation "act_0" {:data v :act-type "sigmoid"})
            (sym/fully-connected "fc_1" {:data v :num-hidden 2})
            (sym/activation "act_1" {:data v :act-type "sigmoid"})
            (sym/linear-regression-output "softmax" {:data v :grad-scale 2}))

        mod (m/module x)]
    (m/bind mod {:data-shapes (mx-io/provide-data-desc train-data) :label-shapes (mx-io/provide-label train-data)})

    (let [arg-params-correct {"fc_0_weight" (ndarray/array [0.15 0.2 0.25 0.3] [2 2])
                              "fc_0_bias" (ndarray/array [0.35 0.35] [2])
                              "fc_1_weight" (ndarray/array [0.4 0.45 05 0.55] [2 2])
                              "fc_1_bias" (ndarray/array [0.6 0.6] [2])}
          arg-params-missing {"fc_0_weight" (ndarray/array [0.15 0.2 0.25 0.3] [2 2])
                              "fc_0_bias" (ndarray/array [0.35 0.35] [2])
                              "fc_1_weight" (ndarray/array [0.4 0.45 05 0.55] [2 2])}
          arg-params-extra {"fc_0_weight" (ndarray/array [0.15 0.2 0.25 0.3] [2 2])
                            "fc_0_bias" (ndarray/array [0.35 0.35] [2])
                            "fc_1_weight" (ndarray/array [0.4 0.45 05 0.55] [2 2])
                            "fc_1_bias" (ndarray/array [0.6 0.6] [2])
                            "fc_2_weight" (ndarray/array [0.6 0.6] [2])}]
      (m/set-params mod {:arg-params arg-params-correct :force-init true})
      (m/set-params mod {:arg-params arg-params-missing :allow-missing true})
      (m/set-params mod {:arg-params arg-params-extra :allow-extra true}))))

(deftest test-monitor
  (let [data (ndarray/array [0.05 0.1] [1 1 1 2])
        label (ndarray/array [0.01 0.99] [1 1 1 2])
        train-data (mx-io/ndarray-iter [data] {:label [label] :label-name "softmax_label"})
        x (as-> (sym/variable "data") v
            (sym/fully-connected "fc_0" {:data v :num-hidden 2})
            (sym/activation "act_0" {:data v :act-type "sigmoid"})
            (sym/fully-connected "fc_1" {:data v :num-hidden 2})
            (sym/activation "act_1" {:data v :act-type "sigmoid"})
            (sym/linear-regression-output "softmax" {:data v :grad-scale 2}))
        ;; create monitor
        mon (monitor/monitor 1 (fn [x]
                                 (ndarray/div (ndarray/sum (ndarray/abs x))
                                              (mx-shape/product (ndarray/shape x)))))
        mod (m/module x {:contexts [(context/cpu 0)]})
        arg-params {"fc_0_weight" (ndarray/array [0.15 0.2 0.25 0.3] [2 2])
                    "fc_0_bias" (ndarray/array [0.35 0.35] [2])
                    "fc_1_weight" (ndarray/array [0.4 0.45 05 0.55] [2 2])
                    "fc_1_bias" (ndarray/array [0.6 0.6] [2])}
        data-batch (mx-io/next train-data)]
    (-> mod
        (m/bind {:data-shapes [{:name "data", :shape [1 1 1 2]}]
                 :label-shapes [{:name "softmax_label", :shape [1 1 1 2]}]})
        (m/install-monitor mon)
        (m/init-params {:arg-params arg-params}))
    (monitor/tic mon)
    (m/forward-backward mod data-batch)
    (let [result (monitor/toc mon)
          freq (->> result
                    (map (fn [v] (as-> (second v) ?
                                   (clojure.string/split ? #"_")
                                   (take 2 ?)
                                   (clojure.string/join "_" ?))))
                    (frequencies))
          expected-freq {"act_0" 2 "act_1" 2 "data" 1 "fc_0" 6 "fc_1" 6}]
      (is (= expected-freq (select-keys freq (keys expected-freq)))))))

(deftest test-forward-reshape
  (let [num-class 10
        data1 (sym/variable "data1")
        data2 (sym/variable "data2")
        conv1 (sym/convolution {:data data1 :kernel [2 2] :num-filter 2 :stride [2 2]})
        conv2 (sym/convolution {:data data2 :kernel [3 3] :num-filter 3 :stride [1 1]})
        pooling1 (sym/pooling {:data conv1 :kernel [2 2] :pool-type "avg" :stride [1 1]})
        pooling2 (sym/pooling {:data conv2 :kernel [2 2] :pool-type "max" :stride [1 1]})
        flatten1 (sym/flatten {:data pooling1})
        flatten2 (sym/flatten {:data pooling2})
        sum (sym/+ (sym/sum {:data flatten1 :axis 1})
                   (sym/sum {:data flatten2 :axis 1}))
        fc (sym/fully-connected {:data sum :num-hidden num-class})
        my-sym (sym/softmax-output "softmax" {:data fc})

        d-shape1 [10 3 64 64]
        d-shape2 [10 3 32 32]
        l-shape [10]
        mod (m/module my-sym {:data-names ["data1" "data2"]})
        data-batch {:data [(ndarray/random-uniform 0 9 (str (mx-shape/->shape d-shape1)))
                           (ndarray/random-uniform 5 15 (str (mx-shape/->shape d-shape2)))]
                    :label [(ndarray/ones l-shape)]
                    :index nil
                    :pad 0}]

   ;; train with the original shapes
    (-> mod
        (m/bind {:data-shapes [{:name "data1" :shape d-shape1}
                               {:name "data2" :shape d-shape2}]
                 :label-shapes [{:name "softmax_label" :shape l-shape :layout "N"}]})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
        (m/forward data-batch))
    (is (= [(first l-shape) num-class]) (-> (m/outputs-merged mod) first (ndarray/shape) (mx-shape/->vec)))
    (-> mod
        (m/backward)
        (m/update))

    (let [d-shape1 [3 3 64 64]
          d-shape2 [3 3 32 32]
          l-shape [3]
          data-batch-2 {:data [(ndarray/random-uniform 0 9 (str (mx-shape/->shape d-shape1)))
                               (ndarray/random-uniform 5 15 (str (mx-shape/->shape d-shape2)))]
                        :label [(ndarray/ones l-shape)]
                        :index nil
                        :pad 0}]
      (-> mod
          (m/forward data-batch))
      (is (= [(first l-shape) num-class]) (-> (m/outputs-merged mod) first (ndarray/shape) (mx-shape/->vec)))
      (-> mod
          (m/backward)
          (m/update)))

    (let [d-shape1 [20 3 64 64]
          d-shape2 [20 3 32 32]
          l-shape [20]
          data-batch-2 {:data [(ndarray/random-uniform 3 5 (str (mx-shape/->shape d-shape1)))
                               (ndarray/random-uniform 10 25 (str (mx-shape/->shape d-shape2)))]
                        :label [(ndarray/ones l-shape)]
                        :index nil
                        :pad 0}]
      (-> mod
          (m/forward data-batch))
      (is (= [(first l-shape) num-class]) (-> (m/outputs-merged mod) first (ndarray/shape) (mx-shape/->vec)))
      (-> mod
          (m/backward)
          (m/update)))

    ;; train with both different batch sizes and data shapes
    (let [d-shape1 [20 3 120 120]
          d-shape2 [20 3 32 64]
          l-shape [20]
          data-batch {:data [(ndarray/random-uniform 0 9 (str (mx-shape/->shape d-shape1)))
                             (ndarray/random-uniform 15 25 (str (mx-shape/->shape d-shape2)))]
                      :label [(ndarray/ones l-shape)]
                      :index nil
                      :pad 0}]
      (-> mod
          (m/forward data-batch))
      (is (= [(first l-shape) num-class]) (-> (m/outputs-merged mod) first (ndarray/shape) (mx-shape/->vec)))
      (-> mod
          (m/backward)
          (m/update)))
    (let [d-shape1 [5 3 28 40]
          d-shape2 [5 3 24 16]
          l-shape [5]
          data-batch {:data [(ndarray/random-uniform 0 9 (str (mx-shape/->shape d-shape1)))
                             (ndarray/random-uniform 15 25 (str (mx-shape/->shape d-shape2)))]
                      :label [(ndarray/ones l-shape)]
                      :index nil
                      :pad 0}]
      (-> mod
          (m/forward data-batch))
      (is (= [(first l-shape) num-class]) (-> (m/outputs-merged mod) first (ndarray/shape) (mx-shape/->vec)))
      (-> mod
          (m/backward)
          (m/update)))))

(comment

  (m/data-shapes x))
