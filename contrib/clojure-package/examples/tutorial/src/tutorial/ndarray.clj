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

(ns tutorial.ndarray
  "A REPL tutorial of the MXNet Clojure API for NDArray, based on
  https://mxnet.incubator.apache.org/api/clojure/ndarray.html"
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]))

;; The NDArray API contains tensor operations similar to
;; `numpy.ndarray`. The syntax is also similar, except for some
;; additional calls for dealing with I/O and multiple devices.


;;;; Create NDArray

;; Create an MXNet NDArray as follows:
(def a (ndarray/zeros [100 50]))            ; all-zero array of dimension 100 x 50
(def b (ndarray/ones [256 32 128 1]))       ; all-one array of given dimensions
(def c (ndarray/array [1 2 3 4 5 6] [2 3])) ; array with given contents in shape 2 x 3

;;; There are also ways to convert an NDArray to a vec or to get the
;;; shape as an object or vec:
(ndarray/->vec c) ;=> [1.0 2.0 3.0 4.0 5.0 6.0]
(ndarray/shape c) ;=> #object[org.apache.mxnet.Shape 0x583c865 "(2,3)"]
(ndarray/shape-vec c) ;=> [2 3]



;; There are some basic NDArray operations, like arithmetic and slice
;; operations.


;;;; NDArray Operations: Arithmetic

(def a (ndarray/ones [1 5]))
(def b (ndarray/ones [1 5]))
(ndarray/->vec (ndarray/+ a b)) ;=>  [2.0 2.0 2.0 2.0 2.0]

;; original ndarrays are unchanged
(ndarray/->vec a) ;=> [1.0 1.0 1.0 1.0 1.0]
(ndarray/->vec b) ;=> [1.0 1.0 1.0 1.0 1.0]

;; inplace operators
(ndarray/+= a b)
(ndarray/->vec a) ;=>  [2.0 2.0 2.0 2.0 2.0]

;; Other arthimetic operations are similar.


;;;; NDArray Operations: Slice

(def a (ndarray/array [1 2 3 4 5 6] [3 2]))
(def a1 (ndarray/slice a 1))
(ndarray/shape-vec a1) ;=> [1 2]
(ndarray/->vec a1) ;=> [3.0 4.0]

(def a2 (ndarray/slice a 1 3))
(ndarray/shape-vec a2) ;=>[2 2]
(ndarray/->vec a2) ;=> [3.0 4.0 5.0 6.0]


;;;; NDArray Operations: Dot Product

(def arr1 (ndarray/array [1 2] [1 2]))
(def arr2 (ndarray/array [3 4] [2 1]))
(def res (ndarray/dot arr1 arr2))
(ndarray/shape-vec res) ;=> [1 1]
(ndarray/->vec res) ;=> [11.0]


;;;; Save and Load NDArray

;; You can use MXNet functions to save and load a map of NDArrays from
;; file systems, as follows:

(ndarray/save "filename" {"arr1" arr1 "arr2" arr2})
;; (you can also do "s3://path" or "hdfs")

(ndarray/save "/Users/daveliepmann/src/coursework/mxnet-clj-tutorials/abc"
              {"arr1" arr1 "arr2" arr2})

;; To load:
(def from-file (ndarray/load "filename"))

from-file ;=>{"arr1" #object[org.apache.mxnet.NDArray 0x6115ba61 "org.apache.mxnet.NDArray@43d85753"], "arr2" #object[org.apache.mxnet.NDArray 0x374b5eff "org.apache.mxnet.NDArray@5c93def4"]}

;; The good thing about using the `save` and `load` interface is that
;; you can use the format across all `mxnet` language bindings. They
;; also already support Amazon S3 and HDFS.


;;;; Multi-Device Support

;; Device information is stored in the `mxnet.Context` structure. When
;; creating NDArray in MXNet, you can use the context argument (the
;; default is the CPU context) to create arrays on specific devices as
;; follows:

(def cpu-a (ndarray/zeros [100 200]))
(ndarray/context cpu-a) ;=> #object[org.apache.mxnet.Context 0x3f376123 "cpu(0)"]

(def gpu-b (ndarray/zeros [100 200] {:ctx (context/gpu 0)})) ;; to use with gpu

;; Currently, we do not allow operations among arrays from different
;; contexts. To manually enable this, use the `copy-to` function to
;; copy the content to different devices, and continue computation.
