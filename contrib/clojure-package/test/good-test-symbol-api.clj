(ns
  ^{:doc "Experimental"}
  org.apache.clojure-mxnet.symbol-api
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
 activation
 "Applies an activation function element-wise to the input.
  
  The following activation functions are supported:
  
  - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
  - `sigmoid`: :math:`y = \\frac{1}{1 + exp(-x)}`
  - `tanh`: Hyperbolic tangent, :math:`y = \\frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
  - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
  - `softsign`: :math:`y = \\frac{x}{1 + abs(x)}`
  
  
  
  Defined in src/operator/nn/activation.cc:L167
  
  `data`: The input array. (optional)
  `act-type`: Activation function to be applied.
  `name`: Name of the symbol (optional)
  `attr`: Attributes of the symbol (optional)"
 [{:keys [data act-type name attr],
   :or {data nil, name nil, attr nil},
   :as opts}]
 (util/coerce-return
  (SymbolAPI/Activation
   (util/->option data)
   act-type
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
 batch-norm
 "Batch normalization.
  
  Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
  well as offset ``beta``.
  
  Assume the input has more than one dimension and we normalize along axis 1.
  We first compute the mean and variance along this axis:
  
  .. math::
  
    data\\_mean[i] = mean(data[:,i,:,...]) \\\\
    data\\_var[i] = var(data[:,i,:,...])
  
  Then compute the normalized output, which has the same shape as input, as following:
  
  .. math::
  
    out[:,i,:,...] = \\frac{data[:,i,:,...] - data\\_mean[i]}{\\sqrt{data\\_var[i]+\\epsilon}} * gamma[i] + beta[i]
  
  Both *mean* and *var* returns a scalar by treating the input as a vector.
  
  Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
  two outputs are blocked.
  
  Besides the inputs and the outputs, this operator accepts two auxiliary
  states, ``moving_mean`` and ``moving_var``, which are *k*-length
  vectors. They are global statistics for the whole dataset, which are updated
  by::
  
    moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
    moving_var = moving_var * momentum + data_var * (1 - momentum)
  
  If ``use_global_stats`` is set to be true, then ``moving_mean`` and
  ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
  the output. It is often used during inference.
  
  The parameter ``axis`` specifies which axis of the input shape denotes
  the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
  axis to be the last item in the input shape.
  
  Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
  then set ``gamma`` to 1 and its gradient to 0.
  
  .. Note::
    When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,
    the sparse tensors will fallback.
  
  
  
  Defined in src/operator/nn/batch_norm.cc:L572
  
  `data`: Input data to batch normalization (optional)
  `gamma`: gamma array (optional)
  `beta`: beta array (optional)
  `moving-mean`: running mean of input (optional)
  `moving-var`: running variance of input (optional)
  `eps`: Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5) (optional)
  `momentum`: Momentum for moving average (optional)
  `fix-gamma`: Fix gamma while training (optional)
  `use-global-stats`: Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator. (optional)
  `output-mean-var`: Output the mean and inverse std  (optional)
  `axis`: Specify which shape axis the channel is specified (optional)
  `cudnn-off`: Do not select CUDNN operator, if available (optional)
  `name`: Name of the symbol (optional)
  `attr`: Attributes of the symbol (optional)"
 [{:keys
   [data
    gamma
    beta
    moving-mean
    moving-var
    eps
    momentum
    fix-gamma
    use-global-stats
    output-mean-var
    axis
    cudnn-off
    name
    attr],
   :or
   {output-mean-var nil,
    axis nil,
    cudnn-off nil,
    fix-gamma nil,
    eps nil,
    data nil,
    attr nil,
    beta nil,
    name nil,
    use-global-stats nil,
    moving-mean nil,
    moving-var nil,
    momentum nil,
    gamma nil},
   :as opts}]
 (util/coerce-return
  (SymbolAPI/BatchNorm
   (util/->option data)
   (util/->option gamma)
   (util/->option beta)
   (util/->option moving-mean)
   (util/->option moving-var)
   (util/->option eps)
   (util/->option momentum)
   (util/->option fix-gamma)
   (util/->option use-global-stats)
   (util/->option output-mean-var)
   (util/->option axis)
   (util/->option cudnn-off)
   name
   (clojure.core/when
    attr
    (clojure.core/->>
     attr
     (clojure.core/mapv
      (clojure.core/fn [[k v]] [k (clojure.core/str v)]))
     (clojure.core/into {})
     util/convert-map)))))

