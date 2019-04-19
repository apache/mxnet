(ns
  ^{:doc "Experimental"}
  org.apache.clojure-mxnet.symbol-random-api
  (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max
                            min repeat reverse set sort take to-array empty sin
                            get apply shuffle ref])
  (:require [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.shape :as mx-shape])
  (:import (org.apache.mxnet SymbolAPI)))

;; Do not edit - this is auto-generated

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




(defn
 exponential
 "Draw random samples from an exponential distribution.\n\nSamples are distributed according to an exponential distribution parametrized by *lambda* (rate).\n\nExample::\n\n   exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],\n                                      [ 0.04146638,  0.31715935]]\n\n\nDefined in src/operator/random/sample_op.cc:L137\n\n`lam`: Lambda parameter (rate) of the exponential distribution. (optional)\n`shape`: Shape of the output. (optional)\n`ctx`: Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls. (optional)\n`dtype`: DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None). (optional)\n`name`: Name of the symbol (optional)\n`attr`: Attributes of the symbol (optional)\n"
 [{:keys [lam shape ctx dtype name attr],
   :or {lam nil, shape nil, ctx nil, dtype nil, name nil, attr nil},
   :as opts}]
 (util/coerce-return
  (SymbolAPI/random_exponential
   (util/->option lam)
   (util/->option (clojure.core/when shape (mx-shape/->shape shape)))
   (util/->option ctx)
   (util/->option dtype)
   name
   (clojure.core/when
    attr
    (clojure.core/->>
     attr
     (clojure.core/mapv
      (clojure.core/fn [[k v]] [k (clojure.core/str v)]))
     (clojure.core/into {})
     util/convert-map)))))

(defn
 gamma
 "Draw random samples from a gamma distribution.\n\nSamples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).\n\nExample::\n\n   gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],\n                                            [ 3.91697288,  3.65933681]]\n\n\nDefined in src/operator/random/sample_op.cc:L125\n\n`alpha`: Alpha parameter (shape) of the gamma distribution. (optional)\n`beta`: Beta parameter (scale) of the gamma distribution. (optional)\n`shape`: Shape of the output. (optional)\n`ctx`: Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls. (optional)\n`dtype`: DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None). (optional)\n`name`: Name of the symbol (optional)\n`attr`: Attributes of the symbol (optional)\n"
 [{:keys [alpha beta shape ctx dtype name attr],
   :or
   {alpha nil,
    beta nil,
    shape nil,
    ctx nil,
    dtype nil,
    name nil,
    attr nil},
   :as opts}]
 (util/coerce-return
  (SymbolAPI/random_gamma
   (util/->option alpha)
   (util/->option beta)
   (util/->option (clojure.core/when shape (mx-shape/->shape shape)))
   (util/->option ctx)
   (util/->option dtype)
   name
   (clojure.core/when
    attr
    (clojure.core/->>
     attr
     (clojure.core/mapv
      (clojure.core/fn [[k v]] [k (clojure.core/str v)]))
     (clojure.core/into {})
     util/convert-map)))))

