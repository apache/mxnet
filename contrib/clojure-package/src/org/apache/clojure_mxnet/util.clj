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

(ns org.apache.clojure-mxnet.util
  (:require [clojure.spec.alpha :as s]
            [t6.from-scala.core :refer [$ $$] :as $]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.primitives :as primitives]
            [org.apache.clojure-mxnet.shape :as mx-shape])
  (:import (org.apache.mxnet NDArray)
           (scala Product Tuple2 Tuple3)
           (scala.collection.immutable List IndexedSeq ListMap)
           (scala.collection JavaConversions Map)
           (scala Option)))

(def ndarray-param-coerce {"float" "num"
                           "int" "num"
                           "boolean" "bool"
                           "scala.collection.immutable.Map" "kwargs-map"
                           "scala.collection.Seq" "& nd-array-and-params"
                           "int<>" "vec-of-ints"
                           "float<>" "vec-of-floats"
                           "byte<>" "byte-array"
                           "java.lang.String<>" "vec-or-strings"
                           "org.apache.mxnet.NDArray" "ndarray"
                           "org.apache.mxnet.Symbol" "sym"
                           "org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE" "double-or-float"})

(def symbol-param-coerce {"java.lang.String" "sym-name"
                          "float" "num"
                          "int" "num"
                          "boolean" "bool"
                          "scala.collection.immutable.Map" "kwargs-map"
                          "scala.collection.Seq" "symbol-list"
                          "int<>" "vec-of-ints"
                          "float<>" "vec-of-floats"
                          "byte<>" "byte-array"
                          "java.lang.String<>" "vec-or-strings"
                          "org.apache.mxnet.Symbol" "sym"
                          "java.lang.Object" "object"})

(defn empty-list []
  ($ List/empty))

(defn empty-map []
  ($ Map/empty))

(defn empty-indexed-seq []
  ($ IndexedSeq/empty))

(defn empty-list-map []
  ($ ListMap/empty))

(defn ->option [v]
  ($ Option v))

(defn ->int-option [v]
  (->option (when v (int v))))

(defn option->value [opt]
  ($/view opt))

(defn keyword->snake-case [vals]
  (mapv (fn [v] (if (keyword? v) (string/replace (name v) "-" "_") v)) vals))

(defn convert-tuple [param]
  (apply $/tuple param))

(def tuple-param-names #{"kernel" "stride" "pad" "target-shape" "shape"})

(defn convert-by-shape [param]
  (into {} (mapv (fn [[k v]]
                   [k (if (vector? v) (mx-shape/->shape v) v)])
                 param)))

(defn tuple-convert-by-param-name [param]
  (into {} (mapv (fn [[k v]]
                   (if (or (get tuple-param-names k)
                           (get tuple-param-names (name k)))
                     [k (str (if (vector? v) (mx-shape/->shape v) v))]
                     [k v]))
                 param)))

(def io-param-names #{"input-shape" "data-shape" "label-shape"})

(defn io-convert-by-param-name [param]
  (into {} (mapv (fn [[k v]] (cond
                               (or (get io-param-names k)
                                   (get io-param-names (name k))) [k (str (if (vector? v) (mx-shape/->shape v) v))]
                               (true? v) [k "True"]
                               (false? v) [k "False"]
                               :else [k (str v)]))
                 param)))

(defn convert-map [param]
  (if (empty? param)
    (empty-map)
    (apply $/immutable-map (->> param
                                (into [])
                                flatten
                                keyword->snake-case))))

(defn convert-symbol-map [param]
  (convert-map (tuple-convert-by-param-name param)))

(defn convert-io-map [param]
  (convert-map (io-convert-by-param-name param)))

(defn convert-shape-map [param]
  (convert-map (convert-by-shape param)))

(defn convert-vector [param]
  (apply $/immutable-list param))

(defn vec->set [param]
  (apply $/immutable-set param))

(defn vec->indexed-seq [x]
  (.toIndexedSeq (convert-vector x)))

(defn apply-scala-fn [f args]
  (.apply f args))

(defn coerce-param [param targets]
  (cond
    (and (get targets "scala.collection.immutable.Map") (map? param)) (convert-map param)
    (and (get targets "float") (number? param)) (float param)
    (and (get targets "scala.collection.Seq") (instance? org.apache.mxnet.NDArray param)) ($/immutable-list param)
    (and (get targets "scala.collection.Seq") (instance? org.apache.mxnet.Symbol param)) ($/immutable-list param)
    (and (get targets "scala.collection.Seq") (and (or (vector? param) (seq? param)) (empty? param))) (empty-list)
    (and (get targets "scala.collection.Seq") (or (vector? param) (seq? param))) (apply $/immutable-list param)
    (and (get targets "int<>") (vector? param)) (int-array param)
    (and (get targets "float<>") (vector? param)) (float-array param)
    (and (get targets "java.lang.String<>") (vector? param)) (into-array param)
    (and (get targets "org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE") (instance? Float param)) (primitives/mx-float param)
    (and (get targets "org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE") (number? param)) (primitives/mx-double param)
    :else param))

(defn nil-or-coerce-param [param targets]
  (when param
    (coerce-param param targets)))

(defn scala-map->map
  [^Map m]
  (into {} (JavaConversions/mapAsJavaMap m)))

(defn buffer->vec [b]
  (into [] (JavaConversions/bufferAsJavaList b)))

(defn scala-vector->vec [x]
  (into [] (JavaConversions/asJavaCollection x)))

(defn scala-iterator->seq [x]
  (iterator-seq (JavaConversions/asJavaIterator x)))

(defn tuple->vec [^Product p]
  (->> (.productArity p)
       (range)
       (map #(.productElement p %))
       (into [])))

(defn coerce-return [return-val]
  (cond
    (instance? scala.collection.mutable.ArrayBuffer return-val) (buffer->vec return-val)
    (instance? scala.collection.immutable.Vector return-val) (scala-vector->vec return-val)
    (instance? org.apache.mxnet.NDArrayFuncReturn return-val) (.head return-val)
    (instance? Map return-val) (scala-map->map return-val)
    (instance? Tuple2 return-val) (tuple->vec return-val)
    (instance? Tuple3 return-val) (tuple->vec return-val)
    (primitives/primitive? return-val) (primitives/->num return-val)
    :else return-val))

(defn coerce-return-recursive [return-val]
  (let [coerced-val (coerce-return return-val)]
    (if (vector? coerced-val)
      (into [] (map coerce-return-recursive coerced-val))
      coerced-val)))

(defmacro scala-fn
  "Creates a scala fn from an anonymous clojure fn of the form (fn [x] body)"
  [f]
  `($/fn ~@(drop-last (rest f)) ~(last f)))

(defn translate-keyword-shape [[k v]]
  [(if (keyword? k) (string/replace (name k) "-" "_") k)
   (if (vector? v) (mx-shape/->shape v) v)])

(defn map->tuple [m]
  (->> m
       (into [])
       (map translate-keyword-shape)
       (map convert-tuple)))

(defn list-map [m]
  (loop [lm ($ ListMap/empty)
         tuples (map->tuple m)]
    (if (seq tuples)
      (recur ($ lm "+" (first tuples)) (rest tuples))
      lm)))

(defn validate! [spec value error-msg]
  (when-not (s/valid? spec value)
    (s/explain spec value)
    (throw (ex-info error-msg
                    (s/explain-data spec value)))))

(defn map->scala-tuple-seq
  "* Convert a map to a scala-Seq of scala-Tubple.
   * Should also work if a seq of seq of 2 things passed.
   * Otherwise passed through unchanged."
  [map-or-tuple-seq]
  (letfn [(key->name [k]
            (if (or (keyword? k) (string? k) (symbol? k))
              (string/replace (name k) "-" "_")
              k))
          (->tuple [kvp-or-tuple]
            (if (coll? kvp-or-tuple)
              (let [[k v] kvp-or-tuple]
                ($/tuple (key->name k) v))
              ;; pass-through
              kvp-or-tuple))]
    (if (coll? map-or-tuple-seq)
      (->> map-or-tuple-seq
           (map ->tuple)
           (apply $/immutable-list))
      ;; pass-through
      map-or-tuple-seq)))
