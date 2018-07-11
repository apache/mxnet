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
  (:require [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]))

;; How to compose symbols
;;The symbolic API provides a way to configure computation graphs. You can configure the graphs either at the level of neural network layer operations or as fine-grained operations.

;;The following example configures a two-layer neural network.

(def data (sym/variable "data"))
(def fc1 (sym/fully-connected "fc1" {:data data :num-hidden 128}))
(def act1 (sym/activation "act1" {:data fc1 :act-type "relu"}))
(def fc2 (sym/fully-connected "fc2" {:data act1 :num-hidden 64}))
(def net (sym/softmax-output "out" {:data fc2}))

;; you could also combine this more dynamically with
(as-> (sym/variable "data") data
  (sym/fully-connected "fc1" {:data data :num-hidden 128})
  (sym/activation "act1" {:data data :act-type "relu"})
  (sym/fully-connected "fc2" {:data data :num-hidden 64})
  (sym/softmax-output "out" {:data data}))

net ;=> #object[org.apache.mxnet.Symbol 0x5c78c8c2 "org.apache.mxnet.Symbol@5c78c8c2"] 


;;The basic arithmetic operators (plus, minus, div, multiplication)

;;The following example creates a computation graph that adds two inputs together.

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))


;; Each symbol takes a (unique) string name. NDArray and Symbol both represent a single tensor. Operators represent the computation between tensors. Operators take symbol (or NDArray) as inputs and might also additionally accept other hyperparameters such as the number of hidden neurons (num_hidden) or the activation type (act_type) and produce the output.

;; We can view a symbol simply as a function taking several arguments. And we can retrieve those arguments with the following method call:

;;We can view a symbol simply as a function taking several arguments. And we can retrieve those arguments with the following method call:

(sym/list-arguments net)
                                        ;=> ["data" "fc1_weight" "fc1_bias" "fc2_weight" "fc2_bias" "out_label"]

;; These arguments are the parameters and inputs needed by each symbol:

;; data: Input data needed by the variable data.
;; fc1_weight and fc1_bias: The weight and bias for the first fully connected layer fc1.
;; fc2_weight and fc2_bias: The weight and bias for the second fully connected layer fc2.
;; out_label: The label needed by the loss.

;;We can also specify the names explicitly:
(def net (sym/variable "data"))
(def w (sym/variable "myweight"))
(def net (sym/fully-connected "fc1" {:data net :weight w :num-hidden 128}))

(sym/list-arguments net)
                                        ;=> ["data" "fc1_weight" "fc1_bias" "fc2_weight" "fc2_bias" "out_label" "myweight" "fc1_bias"]


;;In the above example, FullyConnected layer has 3 inputs: data, weight, bias. When any input is not specified, a variable will be automatically generated for it.


;; More complicated composition

;;MXNet provides well-optimized symbols for layers commonly used in deep learning (see src/operator). We can also define new operators in Python. The following example first performs an element-wise add between two symbols, then feeds them to the fully connected operator:

(def lhs (sym/variable "data1"))
(def rhs (sym/variable "data2"))
(def net (sym/fully-connected "fc1" {:data (sym/+ lhs rhs) :num-hidden 128}))
(sym/list-arguments net) ;=> ["data1" "data2" "fc1_weight" "fc1_bias"]

;; Group Multiple Symbols
;;To construct neural networks with multiple loss layers, we can use mxnet.sym.Group to group multiple symbols together. The following example groups two outputs:

(def net (sym/variable "data"))
(def fc1 (sym/fully-connected {:data net :num-hidden 128}))
(def net2 (sym/activation {:data fc1 :act-type "relu"}))
(def out1 (sym/softmax-output {:data net2}))
(def out2 (sym/linear-regression-output {:data net2}))
(def group (sym/group [out1 out2]))
(sym/list-outputs group);=> ["softmaxoutput0_output" "linearregressionoutput0_output"]


;; Symbol Manipulation
;; One important difference of Symbol compared to NDArray is that we first declare the computation and then bind the computation with data to run.

;; In this section, we introduce the functions to manipulate a symbol directly. But note that, most of them are wrapped by the module package.

;; Shape and Type Inference
;; For each symbol, we can query its arguments, auxiliary states and outputs. We can also infer the output shape and type of the symbol given the known input shape or type of some arguments, which facilitates memory allocation.
(sym/list-arguments fc1) ;=> ["data" "fullyconnected1_weight" "fullyconnected1_bias"]
(sym/list-outputs fc1) ;=> ["fullyconnected1_output"]

;; infer the  shapes given the shape of the input arguments
(let [[arg-shapes out-shapes] (sym/infer-shape fc1 {:data [2 1]})]
  {:arg-shapes arg-shapes
   :out-shapes out-shapes}) ;=> {:arg-shapes ([2 1] [128 1] [128]), :out-shapes ([2 128])}

;; Bind with Data and Evaluate
;; The symbol c constructed above declares what computation should be run. To evaluate it, we first need to feed the arguments, namely free variables, with data.

;; We can do it by using the bind method, which accepts device context and a dict mapping free variable names to NDArrays as arguments and returns an executor. The executor provides forward method for evaluation and an attribute outputs to get all the results.

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))

(def ex (sym/bind c {"a" (ndarray/ones [2 2]) "b" (ndarray/ones [2 2])}))
(-> (executor/forward ex)
    (executor/outputs)
    (first)
    (ndarray/->vec));=>  [2.0 2.0 2.0 2.0]

;;We can evaluate the same symbol on GPU with different data.
;; To do this you must have the correct native library jar defined as a dependency

;;Note In order to execute the following section on a cpu set gpu_device to (cpu).


(def ex (sym/bind c (context/gpu 0) {"a" (ndarray/ones [2 2]) "b" (ndarray/ones [2 2])}))

;; Serialization
;; There are two ways to save and load the symbols. You can use the mxnet.Symbol.save and mxnet.Symbol.load functions to serialize the Symbol objects. The advantage of using save and load functions is that it is language agnostic and cloud friendly. The symbol is saved in JSON format. You can also get a JSON string directly using mxnet.Symbol.toJson. Refer to API documentation for more details.

;; The following example shows how to save a symbol to a file, load it back, and compare two symbols using a JSON string. You can also save to S3 as well

(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))
(sym/save c "symbol-c.json")
(def c2 (sym/load "symbol-c.json"))
(= (sym/to-json c) (sym/to-json c2)) ;=>true

