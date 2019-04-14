(ns
  ^{:doc "Experimental"}
  org.apache.clojure-mxnet.ndarray-api
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:require [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet NDArrayAPI)))

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
 "Applies an activation function element-wise to the input.\n\nThe following activation functions are supported:\n\n- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`\n- `sigmoid`: :math:`y = \\frac{1}{1 + exp(-x)}`\n- `tanh`: Hyperbolic tangent, :math:`y = \\frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`\n- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`\n- `softsign`: :math:`y = \\frac{x}{1 + abs(x)}`\n\n\n\nDefined in src/operator/nn/activation.cc:L167\n\n`data`: The input array.\n`act-type`: Activation function to be applied.\n`out`: Output array. (optional)\n"
 ([data act-type] (activation {:data data, :act-type act-type}))
 ([{:keys [data act-type out], :or {out nil}, :as opts}]
  (util/coerce-return
   (NDArrayAPI/Activation data act-type (util/->option out)))))

(defn
 batch-norm
 "Batch normalization.\n\nNormalizes a data batch by mean and variance, and applies a scale ``gamma`` as\nwell as offset ``beta``.\n\nAssume the input has more than one dimension and we normalize along axis 1.\nWe first compute the mean and variance along this axis:\n\n.. math::\n\n  data\\_mean[i] = mean(data[:,i,:,...]) \\\\\n  data\\_var[i] = var(data[:,i,:,...])\n\nThen compute the normalized output, which has the same shape as input, as following:\n\n.. math::\n\n  out[:,i,:,...] = \\frac{data[:,i,:,...] - data\\_mean[i]}{\\sqrt{data\\_var[i]+\\epsilon}} * gamma[i] + beta[i]\n\nBoth *mean* and *var* returns a scalar by treating the input as a vector.\n\nAssume the input has size *k* on axis 1, then both ``gamma`` and ``beta``\nhave shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and\nthe inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these\ntwo outputs are blocked.\n\nBesides the inputs and the outputs, this operator accepts two auxiliary\nstates, ``moving_mean`` and ``moving_var``, which are *k*-length\nvectors. They are global statistics for the whole dataset, which are updated\nby::\n\n  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)\n  moving_var = moving_var * momentum + data_var * (1 - momentum)\n\nIf ``use_global_stats`` is set to be true, then ``moving_mean`` and\n``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute\nthe output. It is often used during inference.\n\nThe parameter ``axis`` specifies which axis of the input shape denotes\nthe 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel\naxis to be the last item in the input shape.\n\nBoth ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,\nthen set ``gamma`` to 1 and its gradient to 0.\n\n.. Note::\n  When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,\n  the sparse tensors will fallback.\n\n\n\nDefined in src/operator/nn/batch_norm.cc:L574\n\n`data`: Input data to batch normalization\n`gamma`: gamma array\n`beta`: beta array\n`moving-mean`: running mean of input\n`moving-var`: running variance of input\n`eps`: Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5) (optional)\n`momentum`: Momentum for moving average (optional)\n`fix-gamma`: Fix gamma while training (optional)\n`use-global-stats`: Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator. (optional)\n`output-mean-var`: Output the mean and inverse std  (optional)\n`axis`: Specify which shape axis the channel is specified (optional)\n`cudnn-off`: Do not select CUDNN operator, if available (optional)\n`out`: Output array. (optional)\n"
 ([data gamma beta moving-mean moving-var]
  (batch-norm
   {:data data,
    :gamma gamma,
    :beta beta,
    :moving-mean moving-mean,
    :moving-var moving-var}))
 ([{:keys
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
     out],
    :or
    {eps nil,
     momentum nil,
     fix-gamma nil,
     use-global-stats nil,
     output-mean-var nil,
     axis nil,
     cudnn-off nil,
     out nil},
    :as opts}]
  (util/coerce-return
   (NDArrayAPI/BatchNorm
    data
    gamma
    beta
    moving-mean
    moving-var
    (util/->option eps)
    (util/->option momentum)
    (util/->option fix-gamma)
    (util/->option use-global-stats)
    (util/->option output-mean-var)
    (util/->option axis)
    (util/->option cudnn-off)
    (util/->option out)))))

