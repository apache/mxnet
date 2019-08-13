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
  "NDArray API for Clojure package."
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:require
    [clojure.spec.alpha :as s]

    [org.apache.clojure-mxnet.base :as base]
    [org.apache.clojure-mxnet.context :as mx-context]
    [org.apache.clojure-mxnet.shape :as mx-shape]
    [org.apache.clojure-mxnet.util :as util]
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

(defn ->ndarray
  "Creates a new NDArray based on the given n-dimenstional vector
   of numbers.
    `nd-vec`: n-dimensional vector with numbers.
    `opts-map` {
       `ctx`: Context of the output ndarray, will use default context if unspecified.
    }
    returns: `ndarray` with the given values and matching the shape of the input vector.
   Ex:
    (->ndarray [5.0 -4.0])
    (->ndarray [5 -4] {:ctx (context/cpu)})
    (->ndarray [[1 2 3] [4 5 6]])
    (->ndarray [[[1.0] [2.0]]]"
  ([nd-vec {:keys [ctx]
            :or {ctx (mx-context/default-context)}
            :as opts}]
   (array (vec (clojure.core/flatten nd-vec))
          (util/nd-seq-shape nd-vec)
          {:ctx ctx}))
  ([nd-vec] (->ndarray nd-vec {})))

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

(s/def ::ndarray #(instance? NDArray %))
(s/def ::vector vector?)
(s/def ::sequential sequential?)
(s/def ::shape-vec-match-vec
  (fn [[v vec-shape]] (= (count v) (reduce clojure.core/* 1 vec-shape))))

(s/fdef vec->nd-vec
        :args (s/cat :v ::sequential :shape-vec ::sequential)
        :ret ::vector)

(defn- vec->nd-vec
  "Convert a vector `v` into a n-dimensional vector given the `shape-vec`
   Ex:
    (vec->nd-vec [1 2 3] [1 1 3])       ;[[[1 2 3]]]
    (vec->nd-vec [1 2 3 4 5 6] [2 3 1]) ;[[[1] [2] [3]] [[4] [5] [6]]]
    (vec->nd-vec [1 2 3 4 5 6] [1 2 3]) ;[[[1 2 3]] [4 5 6]]]
    (vec->nd-vec [1 2 3 4 5 6] [3 1 2]) ;[[[1 2]] [[3 4]] [[5 6]]]
    (vec->nd-vec [1 2 3 4 5 6] [3 2])   ;[[1 2] [3 4] [5 6]]"
  [v [s1 & ss :as shape-vec]]
  (util/validate! ::sequential v "Invalid input vector `v`")
  (util/validate! ::sequential shape-vec "Invalid input vector `shape-vec`")
  (util/validate! ::shape-vec-match-vec
                  [v shape-vec]
                  "Mismatch between vector `v` and vector `shape-vec`")
  (if-not (seq ss)
    (vec v)
    (->> v
         (partition (clojure.core// (count v) s1))
         vec
         (mapv #(vec->nd-vec % ss)))))

(s/fdef ->nd-vec :args (s/cat :ndarray ::ndarray) :ret ::vector)

(defn ->nd-vec
  "Convert an ndarray `ndarray` into a n-dimensional Clojure vector.
  Ex:
    (->nd-vec (array [1] [1 1 1]))           ;[[[1.0]]]
    (->nd-vec (array [1 2 3] [3 1 1]))       ;[[[1.0]] [[2.0]] [[3.0]]]
    (->nd-vec (array [1 2 3 4 5 6]) [3 1 2]) ;[[[1.0 2.0]] [[3.0 4.0]] [[5.0 6.0]]]"
  [ndarray]
  (util/validate! ::ndarray ndarray "Invalid input array")
  (vec->nd-vec (->vec ndarray) (shape-vec ndarray)))
