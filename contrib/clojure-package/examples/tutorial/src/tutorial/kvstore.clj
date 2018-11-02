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

(ns tutorial.kvstore
  "A REPL tutorial of the MXNet Clojure API for KVStore, based on
  https://mxnet.incubator.apache.org/api/clojure/kvstore.html"
  (:require [org.apache.clojure-mxnet.kvstore :as kvstore]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]))


;;;; Basic Push and Pull

;; Provides basic operation over multiple devices (GPUs or CPUs) on a
;; single device.

;;; Initialization
;; Let’s consider a simple example. It initializes a (`int`,
;; `NDArray`) pair into the store, and then pulls the value out.

(def kv (kvstore/create "local")) ; create a local kvstore
(def shape [2 3])
;; init the kvstore with a vector of keys (strings) and ndarrays
(kvstore/init kv ["3"] [(ndarray/* (ndarray/ones shape) 2)])
(def a (ndarray/zeros shape))
(kvstore/pull kv ["3"] [a])
(ndarray/->vec a) ;=> [2.0 2.0 2.0 2.0 2.0 2.0]


;;; Push, Aggregation, and Updater
;; For any key that’s been initialized, you can push a new value with
;; the same shape to the key, as follows:
(kvstore/push kv ["3"] [(ndarray/* (ndarray/ones shape) 8)])
(kvstore/pull kv ["3"] [a])
(ndarray/->vec a);=>[8.0 8.0 8.0 8.0 8.0 8.0]

;; The data that you want to push can be stored on any
;; device. Furthermore, you can push multiple values into the same
;; key, where KVStore first sums all of these values, and then pushes
;; the aggregated value, as follows:

;; (Here we use multiple CPUs.)
(def cpus [(context/cpu 0) (context/cpu 1) (context/cpu 2)])
(def b [(ndarray/ones shape {:ctx (nth cpus 0)})
        (ndarray/ones shape {:ctx (nth cpus 1)})
        (ndarray/ones shape {:ctx (nth cpus 2)})])
(kvstore/push kv ["3" "3" "3"] b)
(kvstore/pull kv "3" a)
(ndarray/->vec a) ;=> [3.0 3.0 3.0 3.0 3.0 3.0]

;;; Pull
;; You’ve already seen how to pull a single key-value pair. Similar to
;; the way that you use the push command, you can pull the value into
;; several devices with a single call.
(def b [(ndarray/ones shape {:ctx (context/cpu 0)})
        (ndarray/ones shape {:ctx (context/cpu 1)})])
(kvstore/pull kv ["3" "3"] b)
(map ndarray/->vec b) ;=> ([3.0 3.0 3.0 3.0 3.0 3.0] [3.0 3.0 3.0 3.0 3.0 3.0])


;;;; List Key-Value Pairs

;; All of the operations that we’ve discussed so far are performed on
;; a single key. KVStore also provides the interface for generating a
;; list of key-value pairs. For a single device, use the following:

(def ks ["5" "7" "9"])
(kvstore/init kv ks [(ndarray/ones shape)
                     (ndarray/ones shape)
                     (ndarray/ones shape)])
(kvstore/push kv ks [(ndarray/ones shape)
                     (ndarray/ones shape)
                     (ndarray/ones shape)])
(def b [(ndarray/zeros shape)
        (ndarray/zeros shape)
        (ndarray/zeros shape)])
(kvstore/pull kv ks b)
(map ndarray/->vec b) ;=> ([1.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0])


