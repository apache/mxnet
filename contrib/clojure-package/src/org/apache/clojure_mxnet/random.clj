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

(ns org.apache.clojure-mxnet.random
  "Random Number interface of mxnet."
  (:require
   [clojure.spec.alpha :as s]
   [org.apache.clojure-mxnet.context :as context]
   [org.apache.clojure-mxnet.shape :as mx-shape]
   [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet Context Random)))

(s/def ::low number?)
(s/def ::high number?)
(s/def ::low-high (fn [[low high]] (<= low high)))
(s/def ::shape-vec (s/coll-of pos-int? :kind vector?))
(s/def ::ctx #(instance? Context %))
(s/def ::uniform-opts (s/keys :opt-un [::ctx]))

(defn uniform
  "Generate uniform distribution in [`low`, `high`) with shape.
    `low`: The lower bound of distribution.
    `high`: The upper bound of distribution.
    `shape-vec`: vector shape of the ndarray generated.
    `opts-map` {
      `ctx`: Context of output ndarray, will use default context if not specified.
      `out`: Output place holder}
    returns: The result ndarray with generated result.
   Ex:
     (uniform 0 1 [1 10])
     (uniform -10 10 [100 100])"
  ([low high shape-vec {:keys [ctx out] :as opts}]
   (util/validate! ::uniform-opts opts  "Incorrect random uniform parameters")
   (util/validate! ::low low  "Incorrect random uniform parameter")
   (util/validate! ::high high  "Incorrect random uniform parameters")
   (util/validate! ::low-high [low high]  "Incorrect random uniform parameters")
   (util/validate! ::shape-vec shape-vec  "Incorrect random uniform parameters")
   (Random/uniform (float low) (float high) (mx-shape/->shape shape-vec) ctx out))
  ([low high shape-vec]
   (uniform low high shape-vec {})))

(s/def ::loc number?)
(s/def ::scale (s/and number? pos?))
(s/def ::normal-opts (s/keys :opt-un [::ctx]))

(defn normal
  "Generate normal (Gaussian) distribution N(mean, stdvar^^2) with shape.
    `loc`: Mean (centre) of the distribution.
    `scale`: Standard deviation (spread or width) of the distribution.
    `shape-vec`: vector shape of the ndarray generated.
    `opts-map` {
      `ctx`: Context of output ndarray, will use default context if not specified.
      `out`: Output place holder}
    returns: The result ndarray with generated result.
   Ex:
     (normal 0 1 [10 10])
     (normal -5 4 [2 3])"
  ([loc scale shape-vec {:keys [ctx out] :as opts}]
   (util/validate! ::normal-opts opts  "Incorrect random normal parameters")
   (util/validate! ::loc loc  "Incorrect random normal parameters")
   (util/validate! ::scale scale  "Incorrect random normal parameters")
   (util/validate! ::shape-vec shape-vec  "Incorrect random uniform parameters")
   (Random/normal (float loc)
                  (float scale)
                  (mx-shape/->shape shape-vec) ctx out))
  ([loc scale shape-vec]
   (normal loc scale shape-vec {})))

(s/def ::seed-state number?)
(defn seed
  "Seed the random number generators in mxnet.
   This seed will affect behavior of functions in this module,
   as well as results from executors that contains Random number
   such as Dropout operators.

   `seed-state`: The random number seed to set to all devices.
   note: The random number generator of mxnet is by default device specific.
         This means if you set the same seed, the random number sequence
         generated from GPU0 can be different from CPU.
   Ex:
     (seed-state 42)
     (seed-state 42.0)"
  [seed-state]
  (util/validate! ::seed-state seed-state  "Incorrect seed parameters")
  (Random/seed (int seed-state)))
