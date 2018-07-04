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

(ns org.apache.clojure-mxnet.symbol
  (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max
                            min repeat reverse set sort take to-array empty sin
                            get apply shuffle])
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as mx-context]
            [org.apache.clojure-mxnet.executor :as ex]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [t6.from-scala.core :refer [$] :as $]
            [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:import (org.apache.mxnet Symbol)))

;; loads the generated functions into the namespace
(do (clojure.core/load "gen/symbol"))

;;;;;;

(defn variable
  "Create a symbolic variable with a specified name.
  attr-map: Additional attributes to set on the variable
  shape-vec: The shape vector of the variable. If specified, this will be used during shape inference.
  lr-mult: The learning rate multiplier
  wd-mult: The weight decay multiplier for the input variable
  dtype: The dtype for the input variable
  kwarg-map: Additional attributes which must start and end with double underscores"
  ([var-name]
   (variable var-name {}))
  ([var-name  {:keys [attrs shape lr-mult wd-mult dtype kwargs] :as opts}]
   (Symbol/Variable var-name
                    (when attrs (util/convert-symbol-map attrs))
                    (when shape (mx-shape/->shape shape))
                    (if lr-mult (float lr-mult) ($/option nil))
                    (if wd-mult (float wd-mult) ($/option nil))
                    dtype
                    (if kwargs (util/convert-symbol-map kwargs) (util/empty-map)))))

(defn bind
  "Bind the current symbol to get an executor.
  sym: symbol
  ctx: the device context of the generated executor to run on
  bind-map: map of str to ndarray
  bind-grad-map: map of str to ndarray"
  ([sym ctx bind-map-or-vec bind-grads-map-or-vec grad-req bind-aux-map-or-vec]
   (.bind sym
          ctx
          (util/coerce-param bind-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})
          (util/coerce-param bind-grads-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})
          grad-req
          (util/coerce-param bind-aux-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})
          nil
          nil))
  ([sym ctx bind-map-or-vec bind-grads-map-or-vec]
   (.bind sym
          ctx
          (util/coerce-param bind-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})
          (util/coerce-param bind-grads-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})))
  ([sym ctx bind-map-or-vec]
   (.bind sym
          ctx
          (util/coerce-param bind-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"})
          nil))
  ([sym bind-map-or-vec]
   (.bind sym
          (mx-context/default-context)
          (util/coerce-param bind-map-or-vec #{"scala.collection.immutable.Map" "scala.collection.Seq"}))))

(defn simple-bind
  " Bind current symbol to get an executor, allocate all the ndarrays needed.
    Allows specifying data types.
    This function will ask user to pass in ndarray of position
    they like to bind to, and it will automatically allocate the ndarray
    for arguments and auxiliary states that user did not specify explicitly.

   ctx: The device context the generated executor to run on.
   shape-vec-map: map of  name->shape
   opt-map: options map of:
   :grad-req {'write', 'add', 'null'}, or list of str or dict of str to str, optional
                 Specifies how we should update the gradient to the args_grad.
                  - 'write' means everytime gradient is write to specified args_grad NDArray.
                  - 'add' means everytime gradient is add to the specified NDArray.
                  - 'null' means no action is taken, the gradient may not be calculated.
   :type-map map of  name->dtype.
   Will return the generator"
  ([sym ctx shape-vec-map {:keys [grad-req type-map] :as opts
                           :or {grad-req "write"}}]
   (let [shape-map (->> shape-vec-map
                        (map (fn [[k v]] [k (mx-shape/->shape v)]))
                        (into {}))]
     (.simpleBind sym ctx grad-req
                  (util/nil-or-coerce-param shape-map #{"scala.collection.immutable.Map"})
                  (util/nil-or-coerce-param type-map #{"scala.collection.immutable.Map"}))))
  ([sym ctx shape-vec-map]
   (simple-bind sym ctx shape-vec-map {}))
  ([sym ctx]
   (.simpleBind sym ctx "write" (util/empty-map) nil)))

(defn ones
  "Returns a new symbol of given shape and type, filled with ones"
  ([shape-vec {:keys [ctx dtype] :as optss
               :or {ctx nil dtype base/MX_REAL_TYPE}}]
   (Symbol/ones (mx-shape/->shape shape-vec) dtype ctx))
  ([shape-vec]
   (ones shape-vec {})))

(defn zeros
  "Returns a new symbol of given shape and type, filled with zeros"
  ([shape-vec {:keys [ctx dtype] :as opts
               :or {ctx nil dtype base/MX_REAL_TYPE}}]
   (Symbol/zeros (mx-shape/->shape shape-vec) dtype ctx))
  ([shape-vec]
   (zeros shape-vec {})))

(defn arange
  "Returns evenly spaced values within a given interval.
   Values are generated within the half-open interval [`start`, `stop`). In other
   words, the interval includes `start` but excludes `stop`."
  ([start stop  {:keys [step repeat dtype]
                 :or {step (float 1) repeat (int 1) dtype base/MX_REAL_TYPE}
                 :as opts}]
   (Symbol/arange (float start) ($/option (float stop)) step repeat nil dtype))
  ([start stop]
   (arange start stop {})))

;;; manually defined because of a conflicting arity of 2 with the auto-gen
(defn min
  ([sym-name kwargs-map symbol-list kwargs-map-1]
   (util/coerce-return
    (Symbol/min
     (util/nil-or-coerce-param sym-name #{"java.lang.String"})
     (util/nil-or-coerce-param
      kwargs-map
      #{"scala.collection.immutable.Map"})
     (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
     (util/nil-or-coerce-param
      kwargs-map-1
      #{"scala.collection.immutable.Map"}))))
  ([sym-name attr-map kwargs-map]
   (min sym-name attr-map (util/empty-list) kwargs-map))
  ([kwargs-map] (min nil nil (util/empty-list) kwargs-map))
  ([sym1 sym2]
   (util/coerce-return
    (Symbol/min
     (util/nil-or-coerce-param
      sym1
      #{"ml.dmlc.mxnet.Symbol" "java.lang.Object"})
     (util/nil-or-coerce-param
      sym2
      #{"ml.dmlc.mxnet.Symbol" "java.lang.Object"})))))

;;; manually defined because of a conflicting arity of 2 with the auto-gen

(defn max
  ([sym1 sym2]
   (util/coerce-return
    (Symbol/max
     (util/nil-or-coerce-param
      sym1
      #{"ml.dmlc.mxnet.Symbol" "java.lang.Object"})
     (util/nil-or-coerce-param
      sym2
      #{"ml.dmlc.mxnet.Symbol" "java.lang.Object"}))))
  ([sym-name kwargs-map symbol-list kwargs-map-1]
   (util/coerce-return
    (Symbol/max
     (util/nil-or-coerce-param sym-name #{"java.lang.String"})
     (util/nil-or-coerce-param
      kwargs-map
      #{"scala.collection.immutable.Map"})
     (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
     (util/nil-or-coerce-param
      kwargs-map-1
      #{"scala.collection.immutable.Map"}))))
  ([sym-name attr-map kwargs-map]
   (max sym-name attr-map (util/empty-list) kwargs-map))
  ([kwargs-map] (max nil nil (util/empty-list) kwargs-map)))

;;; redefining to make it easier to work with

(defn- coerce-infer-shape-return [ret]
  (->> ret
       (map util/scala-vector->vec)
       (map (fn [shapes] (map mx-shape/->vec shapes)))))

(defn
  infer-shape
  ([sym vec-or-strings vec-of-ints vec-of-ints-1]
   (let [ret (util/coerce-return
              (.inferShape
               sym
               (util/nil-or-coerce-param vec-or-strings #{"java.lang.String<>"})
               (util/nil-or-coerce-param vec-of-ints #{"int<>"})
               (util/nil-or-coerce-param vec-of-ints-1 #{"int<>"})))]
     (coerce-infer-shape-return ret)))
  ([sym symbol-list-or-kwargs-map]
   (let [ret (util/coerce-return
              (.inferShape
               sym
               (if (map? symbol-list-or-kwargs-map)
                 (util/convert-shape-map symbol-list-or-kwargs-map)
                 (util/nil-or-coerce-param
                  symbol-list-or-kwargs-map
                  #{"scala.collection.Seq" "scala.collection.immutable.Map"}))))]
     (coerce-infer-shape-return ret))))

(defn
  save-checkpoint
  "Taken from the model save checkpoint"
  [prefix epoch sym arg-params aux-params]
  (do
    (save sym (str prefix "-symbol.json"))
    (let [save-map (merge (->> arg-params
                               (mapv (fn [[k v]] [(str "arg:" k) v]))
                               (into {}))
                          (->> aux-params
                               (mapv (fn [[k v]] [(str "aux:" k) v]))
                               (into {})))
          param-name (format "%s-%04d.params" prefix epoch)]
      (ndarray/save param-name save-map)
      (println "Saved checkpoint to " param-name))))
