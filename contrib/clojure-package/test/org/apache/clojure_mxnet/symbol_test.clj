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

(ns org.apache.clojure-mxnet.symbol-test
  (:require [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.context :as context]))

(deftest test-compose
  (let [data (sym/variable "data")
        net1 (sym/fully-connected "fc1" {:data data :num-hidden 10})
        net1 (sym/fully-connected "fc2" {:data net1 :num-hidden 100})

        net2 (sym/fully-connected "fc3" {:num-hidden 10})
        net2 (sym/activation {:data net2 :act-type "relu"})
        net2 (sym/fully-connected "fc4" {:data net2 :num-hidden 20})

        composed (sym/apply net2 "composed" {"fc3_data" net1})

        multi-out (sym/group [composed net1])]

    (is (= ["data" "fc1_weight" "fc1_bias" "fc2_weight" "fc2_bias"] (sym/list-arguments net1)))
    (println (sym/debug-str composed))
    (is (= 2 (count (sym/list-outputs multi-out))))))

(deftest test-symbol-internal
  (let [data (sym/variable "data")
        oldfc (sym/fully-connected "fc1" {:data data :num-hidden 10})
        net1 (sym/fully-connected "fc2" {:data oldfc :num-hidden 100})]
    (is (= ["data" "fc1_weight" "fc1_bias" "fc2_weight" "fc2_bias"] (sym/list-arguments net1)))
    (= (sym/list-arguments oldfc) (-> (sym/get-internals net1)
                                      (sym/get "fc1_output")
                                      (sym/list-arguments)))))

(deftest test-infer-type
  (let [data (sym/variable "data")
        f32data (sym/cast {:data data :dtype "float32"})
        fc1 (sym/fully-connected "fc1" {:data f32data :num-hidden 128})
        mlp (sym/softmax-output "softmax" {:data fc1})
        [arg out aux] (sym/infer-type mlp {:data dtype/FLOAT64})]
    (is (= [dtype/FLOAT64 dtype/FLOAT32 dtype/FLOAT32 dtype/FLOAT32] (util/buffer->vec arg)))
    (is (= [dtype/FLOAT32] (util/buffer->vec out)))
    (is (= [] (util/buffer->vec aux)))))

(deftest test-copy
  (let [data (sym/variable "data")
        data2 (sym/clone data)]
    (is (= (sym/to-json data) (sym/to-json data2)))))

(deftest test-basic-bind
  (let [a (sym/variable "a")
        b (sym/variable "b")
        c (sym/+ a b)
        ex (sym/bind c {"a" (ndarray/ones [2 2]) "b" (ndarray/ones [2 2])})]
    (is (= [2.0 2.0 2.0 2.0] (-> (executor/forward ex)
                                 (executor/outputs)
                                 (first)
                                 (ndarray/->vec))))))
(deftest test-simple-bind
  (let [a (sym/ones [3])
        b (sym/ones [3])
        c (sym/+ a b)
        ex (sym/simple-bind c (context/default-context))]
    (is (= [2.0 2.0 2.0]  (-> (executor/forward ex)
                              (executor/outputs)
                              (first)
                              (ndarray/->vec))))))

(deftest test-infer-shape
  (let [a (sym/variable "a")
        b (sym/variable "b")
        c (sym/+ a b)
        [arg-shapes out-shapes] (sym/infer-shape c {"a" [2 2] "b" [2 2]})]
    (is (= [[2 2] [2 2]] arg-shapes))
    (is (= [[2 2]] out-shapes))))
