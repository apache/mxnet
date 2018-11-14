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

(ns org.apache.clojure-mxnet.ndarray
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle])
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as mx-context]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.reflect :as r]
            [t6.from-scala.core :refer [$] :as $])
  (:import (org.apache.mxnet NDArray)))

;; loads the generated functions into the namespace
(do (clojure.core/load "gen/ndarray"))

(defn ->vec
  "Converts a nd-array to a vector (one dimensional)"
  [ndarray]
  (-> ndarray to-array aclone vec))

(defn empty
  "Create an empty uninitialized new NDArray, with specified shape"
  ([shape-vec {:keys [ctx dtype]
               :or {ctx (mx-context/default-context) dtype base/MX_REAL_TYPE}
               :as opts}]
   (NDArray/empty (mx-shape/->shape shape-vec) ctx dtype))
  ([shape-vec]
   (empty shape-vec {})))

(defn zeros
  "Create a new NDArray filled with 0, with specified shape."
  ([shape-vec {:keys [ctx dtype]
               :or {ctx (mx-context/default-context) dtype base/MX_REAL_TYPE}
               :as opts}]
   (NDArray/zeros (mx-shape/->shape shape-vec) ctx dtype))
  ([shape-vec]
   (zeros shape-vec {})))

(defn ones
  "Create a new NDArray filled with 1, with specified shape."
  ([shape-vec {:keys [ctx dtype]
               :or {ctx (mx-context/default-context) dtype base/MX_REAL_TYPE}
               :as opts}]
   (NDArray/ones (mx-shape/->shape shape-vec) ctx dtype))
  ([shape-vec]
   (ones shape-vec {})))

(defn full
  "Create a new NDArray filled with given value, with specified shape."
  ([shape-vec value {:keys [ctx dtype]
                     :or {ctx (mx-context/default-context)}
                     :as opts}]
   (NDArray/full (mx-shape/->shape shape-vec) value ctx))
  ([shape-vec value]
   (full shape-vec value {})))

(defn array
  "Create a new NDArray that copies content from source vector"
  ([source-vec shape-vec {:keys [ctx dtype]
                          :or {ctx (mx-context/default-context)}
                          :as opts}]
   (NDArray/array (float-array source-vec) (mx-shape/->shape shape-vec) ctx))
  ([source-vec shape-vec]
   (array source-vec shape-vec {})))

(defn arange
  "Returns evenly spaced values within a given interval.
   Values are generated within the half-open interval [`start`, `stop`). In other
   words, the interval includes `start` but excludes `stop`."
  ([start stop  {:keys [step repeat ctx dtype]
                 :or {step (float 1) repeat (int 1) ctx (mx-context/default-context) dtype base/MX_REAL_TYPE}
                 :as opts}]
   (NDArray/arange (float start) ($/option (float stop)) step repeat ctx dtype))
  ([start stop]
   (arange start stop {})))

(defn slice
  "Return a sliced NDArray that shares memory with current one."
  ([ndarray i]
   (.slice ndarray (int i)))
  ([ndarray start stop]
   (.slice ndarray (int start) (int stop))))

(defn copy-to
  "Copy the content of current array to other"
  [source-ndarray target-ndarray]
  (.copyTo source-ndarray target-ndarray))

(defn save
  "Save list of NDArray or dict of str->NDArray to binary file
 (The name of the file.Can be S3 or HDFS address (remember built with S3 support))
 Example of fname:
   *     - `s3://my-bucket/path/my-s3-ndarray`
   *     - `hdfs://my-bucket/path/my-hdfs-ndarray`
   *     - `/path-to/my-local-ndarray`"
  [fname map-of-name-to-ndarray]
  (NDArray/save fname (util/coerce-param map-of-name-to-ndarray #{"scala.collection.immutable.Map"})))

(defn load
  "Takes a filename and returns back a map of ndarray-name to ndarray"
  [filename]
  (let [info (NDArray/load filename)
        [names ndarrays] (util/tuple->vec info)]
    (into {} (map (fn [n a] {(str n) a}) names ndarrays))))

(defn save-to-file
  "Save one ndarray to a file"
  [fname ndarray]
  (save fname {"default" ndarray}))

(defn load-from-file
  "Load one ndarry from a file"
  [fname]
  (first (load2-array fname)))

(defn as-in-context
  "Return an `NDArray` that lives in the target context. If the array
   is already in that context, `self` is returned. Otherwise, a copy is made."
  [ndarray ctx]
  (.asInContext ndarray ctx))

(defn as-type
  "Return a copied numpy array of current array with specified type."
  [ndarray dtype]
  (.asType ndarray dtype))

(defn / [ndarray num-or-NDArray]
  (div ndarray num-or-NDArray))

(defn concatenate
  ([ndarrays {:keys [axis always-copy] :or {axis 1 always-copy true}}]
   (NDArray/concatenate (apply $/immutable-list ndarrays) (int axis) always-copy))
  ([ndarrays]
   (NDArray/concatenate (apply $/immutable-list ndarrays))))

(defn ->raw [ndarray]
  (-> ndarray internal .getRaw))

(defn ->float-vec [ndarray]
  (-> ndarray internal .toFloatArray vec))

(defn ->int-vec [ndarray]
  (-> ndarray internal .toIntArray vec))

(defn ->double-vec [ndarray]
  (-> ndarray internal .toDoubleArray vec))

(defn ->byte-vec [ndarray]
  (-> ndarray internal .toByteArray vec))

(defn shape-vec [ndarray]
  (mx-shape/->vec (shape ndarray)))
