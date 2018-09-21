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

(ns org.apache.clojure-mxnet.executor
  (:require [org.apache.clojure-mxnet.util :as util]
            [clojure.reflect :as r]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]))

;; need to revisit to get all functions

(defn ->vec [nd-array]
  (vec (.toArray nd-array)))

(defn forward
  "* Calculate the outputs specified by the binded symbol.
   * @param is-train whether this forward is for evaluation purpose.
   * @param kwargs Additional specification of input arguments."
  ([executor]
   (do (.forward executor)
       executor))
  ([executor is-train kwargs]
   (do (.forward executor is-train (util/map->scala-tuple-seq kwargs))
       executor)))

(defn backward
  "* Do backward pass to get the gradient of arguments.
   * @param ndarray-or-vec Gradient on the outputs to be propagated back.
   *                       This parameter is only needed when bind is called
   *                        on outputs that are not a loss function."
  ([executor]
   (do (.backward executor)
       executor))
  ([executor ndarray-or-vec]
   (do (.backward executor (if (vector? ndarray-or-vec) (into-array ndarray-or-vec) ndarray-or-vec))
       executor)))

(defn outputs [executor]
  "list all the output ndarrays"
  (.outputs executor))

(defn grad-arrays [executor]
  "list all the gradient ndarrays"
  (.gradArrays executor))

(defn arg-arrays [executor]
  "list all the argument ndarrays"
  (.argArrays executor))

(defn grad-map [executor]
  (util/scala-map->map (.gradDict executor)))

(defn arg-map [executor]
  (util/scala-map->map (.argDict executor)))

(defn set-arg [executor arg-name arg-val-or-vec]
  (-> executor
      (arg-map)
      (get arg-name)
      (ndarray/set arg-val-or-vec)))

(defn set-arg-arrays [executor vec-of-ndarray-or-val]
  (doall (map (fn [arg-array v] (ndarray/set arg-array v)) (vec (arg-arrays executor)) vec-of-ndarray-or-val)))

(defn get-grad [executor grad-name]
  (-> executor
      (grad-map)
      (get grad-name)))

(defn reshape
  " * Return a new executor with the same symbol and shared memory,
   * but different input/output shapes.
   * For runtime reshaping, variable length sequences, etc.
   * The returned executor shares state with the current one,
   * and cannot be used in parallel with it.
   * @param kwargs Map of string to shape-vec.
   *                - new shape for arguments.
   * @parms opts with :partial-shaping Whether to allow changing the shape of unspecified arguments.
   * and  :allow-up-sizing Whether to allow allocating new ndarrays that's larger than the original."
  ([executor kwargs {:keys [partial-shaping allow-up-sizing]
                     :or {partial-shaping false allow-up-sizing false}}]
   (do
     (let [kwargs-shapes (zipmap (keys kwargs)
                                 (mapv (fn [v] (if (vector? v) (mx-shape/->shape v) v)) (vals kwargs)))]
       (.reshape executor partial-shaping allow-up-sizing (util/convert-map kwargs-shapes)))
     executor))
  ([executor kwargs]
   (reshape executor kwargs {})))
