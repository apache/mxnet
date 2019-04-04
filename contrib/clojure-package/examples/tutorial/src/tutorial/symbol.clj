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

(ns tutorial.symbol
  "A REPL tutorial of the MXNet Clojure Symbolic API, based on
  https://mxnet.incubator.apache.org/api/clojure/symbol.html"
  (:require [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]))


;;;; How to Compose Symbols

;; The symbolic API provides a way to configure computation
;; graphs. You can configure the graphs either at the level of neural
;; network layer operations or as fine-grained operations.

;; The following example configures a two-layer neural network.
(def data (sym/variable "data"))
(def fc1 (sym/fully-connected "fc1" {:data data :num-hidden 128}))
(def act1 (sym/activation "act1" {:data fc1 :act-type "relu"}))
(def fc2 (sym/fully-connected "fc2" {:data act1 :num-hidden 64}))
(def net (sym/softmax-output "out" {:data fc2}))

;; This can also be combined more dynamically with the `as->` Clojure
;; threading form.
(as-> (sym/variable "data") data
  (sym/fully-connected "fc1" {:data data :num-hidden 128})
  (sym/activation "act1"     {:data data :act-type "relu"})
  (sym/fully-connected "fc2" {:data data :num-hidden 64})
  (sym/softmax-output "out"  {:data data}))

net ;=> #object[org.apache.mxnet.Symbol 0x5c78c8c2 "org.apache.mxnet.Symbol@5c78c8c2"] 

;; The basic arithmetic operators (plus, minus, div, multiplication)
;; work as expected. The following example creates a computation graph
;; that adds two inputs together.

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))


;;;; More Complicated Compositions

;; MXNet provides well-optimized symbols for layers commonly used in
;; deep learning (see src/operator). We can also define new operators
;; in Python. The following example first performs an element-wise add
;; between two symbols, then feeds them to the fully connected
;; operator:

(def lhs (sym/variable "data1"))
(def rhs (sym/variable "data2"))
(def net (sym/fully-connected "fc1" {:data (sym/+ lhs rhs)
                                     :num-hidden 128}))
(sym/list-arguments net) ;=> ["data1" "data2" "fc1_weight" "fc1_bias"]


;;;; Group Multiple Symbols

;; To construct neural networks with multiple loss layers, we can use
;; `group` to group multiple symbols together. The following example
;; groups two outputs:

(def net (sym/variable "data"))
(def fc1 (sym/fully-connected {:data net :num-hidden 128}))
(def net2 (sym/activation {:data fc1 :act-type "relu"}))
(def out1 (sym/softmax-output {:data net2}))
(def out2 (sym/linear-regression-output {:data net2}))
(def group (sym/group [out1 out2]))
(sym/list-outputs group) ;=> ["softmaxoutput0_output" "linearregressionoutput0_output"]


;;;; Serialization

;; You can use the `save` and `load` functions to serialize Symbol
;; objects as JSON. These functions have the advantage of being
;; language-agnostic and cloud-friendly. You can also get a JSON
;; string directly using `to-json`.

;; The following example shows how to save a symbol to a file, load it
;; back, and compare two symbols using a JSON string. You can also
;; save to S3 as well.

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))
(sym/save c "symbol-c.json")
(def c2 (sym/load "symbol-c.json"))
(= (sym/to-json c) (sym/to-json c2)) ;=>true


;;;; Executing Symbols

;; To execute symbols, first we need to define the data that they
;; should run on. We can do this with the `bind` function, which
;; returns an executor. We then use `forward` to evaluate and
;; `outputs` to get the results.

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))

(def ex
  (sym/bind c {"a" (ndarray/ones [2 2])
               "b" (ndarray/ones [2 2])}))

(-> (executor/forward ex)
    (executor/outputs)
    (first)
    (ndarray/->vec));=>  [2.0 2.0 2.0 2.0]

;; We can evaluate the same symbol on GPU with different data.
;; (To do this you must have the correct native library jar defined as a dependency.)
(def ex (sym/bind c (context/gpu 0) {"a" (ndarray/ones [2 2])
                                     "b" (ndarray/ones [2 2])}))
