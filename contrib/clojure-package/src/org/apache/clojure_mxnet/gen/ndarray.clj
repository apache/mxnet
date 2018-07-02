(ns org.apache.clojure-mxnet.ndarray
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle])
  (:import (org.apache.mxnet NDArray Shape)))

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
 div
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$div
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 div=
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$div$eq
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 >
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$greater
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 >=
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$greater$eq
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 <
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$less
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 <=
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$less$eq
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 -
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$minus
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 -=
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$minus$eq
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 %
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$percent
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 %=
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$percent$eq
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 +
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$plus
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 +=
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$plus$eq
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 *
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$times
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 *=
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$times$eq
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 **
 ([ndarray num-or-ndarray]
  (util/coerce-return
   (.$times$times
    ndarray
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 **=
 ([ndarray ndarray-or-num]
  (util/coerce-return
   (.$times$times$eq
    ndarray
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 activation
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Activation
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 batch-norm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/BatchNorm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 batch-norm-v1
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/BatchNorm_v1
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 bilinear-sampler
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/BilinearSampler
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 block-grad
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/BlockGrad
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 cast
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Cast
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 concat
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Concat
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 convolution
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Convolution
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 convolution-v1
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Convolution_v1
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 correlation
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Correlation
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 crop
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Crop
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 custom
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Custom
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 deconvolution
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Deconvolution
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 dropout
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Dropout
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 element-wise-sum
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ElementWiseSum
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 embedding
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Embedding
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 flatten
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Flatten
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 fully-connected
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/FullyConnected
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 grid-generator
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/GridGenerator
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 identity-attach-kl-sparse-reg
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/IdentityAttachKLSparseReg
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 instance-norm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/InstanceNorm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 l2-normalization
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/L2Normalization
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 lrn
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/LRN
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 layer-norm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/LayerNorm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 leaky-re-lu
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/LeakyReLU
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linear-regression-output
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/LinearRegressionOutput
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 logistic-regression-output
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/LogisticRegressionOutput
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 mae-regression-output
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/MAERegressionOutput
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 make-loss
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/MakeLoss
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 pad
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Pad
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 pooling
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Pooling
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 pooling-v1
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Pooling_v1
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rnn
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/RNN
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 roi-pooling
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ROIPooling
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 reshape
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Reshape
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 svm-output
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SVMOutput
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sequence-last
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SequenceLast
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sequence-mask
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SequenceMask
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sequence-reverse
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SequenceReverse
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 slice-channel
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SliceChannel
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softmax
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/Softmax
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softmax-activation
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SoftmaxActivation
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softmax-output
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SoftmaxOutput
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 spatial-transformer
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SpatialTransformer
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 swap-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/SwapAxis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn t ([ndarray] (util/coerce-return (.T ndarray))))

(defn
 up-sampling
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/UpSampling
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 abs
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/abs
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 adam-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/adam_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 add-n
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/add_n
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arccos
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arccos
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arccosh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arccosh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arcsin
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arcsin
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arcsinh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arcsinh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arctan
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arctan
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 arctanh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/arctanh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 argmax
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/argmax
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 argmax-channel
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/argmax_channel
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 argmin
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/argmin
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 argsort
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/argsort
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 at
 ([ndarray num]
  (util/coerce-return (.at ndarray (util/coerce-param num #{"int"})))))

(defn
 batch-dot
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/batch_dot
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 batch-take
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/batch_take
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-add
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_add
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-axes
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_axes
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_axis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-div
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_div
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-equal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_equal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-greater
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_greater
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-greater-equal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_greater_equal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-hypot
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_hypot
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-lesser
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_lesser
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-lesser-equal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_lesser_equal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-maximum
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_maximum
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-minimum
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_minimum
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-minus
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_minus
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-mod
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_mod
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-mul
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_mul
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-not-equal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_not_equal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-plus
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_plus
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-power
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_power
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-sub
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_sub
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 broadcast-to
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/broadcast_to
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 cast
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/cast
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 cast-storage
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/cast_storage
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 cbrt
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/cbrt
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 ceil
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ceil
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 choose-element-0index
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/choose_element_0index
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 clip
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/clip
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 concat
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/concat
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 concatenate
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/concatenate
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn context ([ndarray] (util/coerce-return (.context ndarray))))

(defn copy ([ndarray] (util/coerce-return (.copy ndarray))))

(defn
 cos
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/cos
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 cosh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/cosh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 creation-trace
 ([ndarray] (util/coerce-return (.creationTrace ndarray))))

(defn
 crop
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/crop
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 degrees
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/degrees
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 dependencies
 ([ndarray] (util/coerce-return (.dependencies ndarray))))

(defn
 deserialize
 ([byte-array]
  (util/coerce-return
   (NDArray/deserialize (util/coerce-param byte-array #{"byte<>"})))))

(defn dispose ([ndarray] (util/coerce-return (.dispose ndarray))))

(defn
 dispose-deps
 ([ndarray] (util/coerce-return (.disposeDeps ndarray))))

(defn
 dispose-deps-except
 ([ndarray & nd-array-and-params]
  (util/coerce-return
   (.disposeDepsExcept
    ndarray
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 dot
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/dot
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn dtype ([ndarray] (util/coerce-return (.dtype ndarray))))

(defn
 elemwise-add
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/elemwise_add
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 elemwise-div
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/elemwise_div
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 elemwise-mul
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/elemwise_mul
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 elemwise-sub
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/elemwise_sub
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 equal
 ([ndarray-or-ndarray ndarray-or-num]
  (util/coerce-return
   (NDArray/equal
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 equals
 ([ndarray Object]
  (util/coerce-return
   (.equals ndarray (util/coerce-param Object #{"java.lang.Object"})))))

(defn
 exp
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/exp
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 expand-dims
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/expand_dims
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 expm1
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/expm1
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 fill-element-0index
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/fill_element_0index
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn finalize ([ndarray] (util/coerce-return (.finalize ndarray))))

(defn
 fix
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/fix
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 flatten
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/flatten
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 flip
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/flip
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 floor
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/floor
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 ftml-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ftml_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 ftrl-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ftrl_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 gamma
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/gamma
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 gammaln
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/gammaln
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 gather-nd
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/gather_nd
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 greater
 ([ndarray-or-ndarray ndarray-or-num]
  (util/coerce-return
   (NDArray/greater
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 greater-equal
 ([ndarray-or-ndarray num-or-ndarray]
  (util/coerce-return
   (NDArray/greaterEqual
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn handle ([ndarray] (util/coerce-return (.handle ndarray))))

(defn hash-code ([ndarray] (util/coerce-return (.hashCode ndarray))))

(defn
 identity
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/identity
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn internal ([ndarray] (util/coerce-return (.internal ndarray))))

(defn
 is-disposed
 ([ndarray] (util/coerce-return (.isDisposed ndarray))))

(defn
 khatri-rao
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/khatri_rao
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 lesser
 ([ndarray-or-ndarray ndarray-or-num]
  (util/coerce-return
   (NDArray/lesser
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 lesser-equal
 ([ndarray-or-ndarray ndarray-or-num]
  (util/coerce-return
   (NDArray/lesserEqual
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 linalg-gelqf
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_gelqf
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-gemm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_gemm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-gemm2
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_gemm2
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-potrf
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_potrf
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-potri
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_potri
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-sumlogdiag
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_sumlogdiag
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-syrk
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_syrk
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-trmm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_trmm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 linalg-trsm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/linalg_trsm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 load
 ([String]
  (util/coerce-return
   (NDArray/load (util/coerce-param String #{"java.lang.String"})))))

(defn
 load2-array
 ([String]
  (util/coerce-return
   (NDArray/load2Array
    (util/coerce-param String #{"java.lang.String"})))))

(defn
 load2-map
 ([String]
  (util/coerce-return
   (NDArray/load2Map
    (util/coerce-param String #{"java.lang.String"})))))

(defn
 log
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/log
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 log10
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/log10
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 log1p
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/log1p
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 log2
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/log2
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 log-dispose-warning
 ([ndarray] (util/coerce-return (.logDisposeWarning ndarray))))

(defn
 log-softmax
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/log_softmax
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 make-loss
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/make_loss
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 max
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/max
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 max-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/max_axis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 maximum
 ([num-or-ndarray-or-ndarray ndarray-or-num-or-ndarray]
  (util/coerce-return
   (NDArray/maximum
    (util/coerce-param
     num-or-ndarray-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 mean
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/mean
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 min
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/min
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 min-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/min_axis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 minimum
 ([ndarray-or-ndarray-or-num num-or-ndarray-or-ndarray]
  (util/coerce-return
   (NDArray/minimum
    (util/coerce-param
     ndarray-or-ndarray-or-num
     #{"float" "org.apache.mxnet.NDArray"})
    (util/coerce-param
     num-or-ndarray-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 mp-sgd-mom-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/mp_sgd_mom_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 mp-sgd-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/mp_sgd_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 nanprod
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/nanprod
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 nansum
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/nansum
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 negative
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/negative
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 norm
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/norm
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 normal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/normal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 not-equal
 ([ndarray-or-ndarray num-or-ndarray]
  (util/coerce-return
   (NDArray/notEqual
    (util/coerce-param
     ndarray-or-ndarray
     #{"org.apache.mxnet.NDArray"})
    (util/coerce-param
     num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 one-hot
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/one_hot
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 onehot-encode
 ([ndarray ndarray-1]
  (util/coerce-return
   (NDArray/onehotEncode
    (util/coerce-param ndarray #{"org.apache.mxnet.NDArray"})
    (util/coerce-param ndarray-1 #{"org.apache.mxnet.NDArray"})))))

(defn
 ones-like
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/ones_like
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 org.apache.mxnet.nd-array
 ([ndarray long bool]
  (util/coerce-return
   (.org.apache.mxnet.NDArray
    ndarray
    (util/coerce-param long #{"long"})
    (util/coerce-param bool #{"boolean"})))))

(defn
 pad
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/pad
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 pick
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/pick
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 power
 ([num-or-ndarray-or-ndarray ndarray-or-num-or-ndarray]
  (util/coerce-return
   (NDArray/power
    (util/coerce-param
     num-or-ndarray-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})
    (util/coerce-param
     ndarray-or-num-or-ndarray
     #{"float" "org.apache.mxnet.NDArray"})))))

(defn
 prod
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/prod
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 radians
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/radians
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-exponential
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_exponential
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-gamma
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_gamma
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-generalized-negative-binomial
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_generalized_negative_binomial
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-negative-binomial
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_negative_binomial
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-normal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_normal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-poisson
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_poisson
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 random-uniform
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/random_uniform
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rcbrt
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/rcbrt
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 reciprocal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/reciprocal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 relu
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/relu
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 repeat
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/repeat
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 reshape
 ([ndarray Shape-or-vec-of-ints]
  (util/coerce-return
   (.reshape
    ndarray
    (util/coerce-param
     Shape-or-vec-of-ints
     #{"org.apache.mxnet.Shape" "int<>"})))))

(defn
 reshape-like
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/reshape_like
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 reverse
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/reverse
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rint
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/rint
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rmsprop-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/rmsprop_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rmspropalex-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/rmspropalex_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 round
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/round
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 rsqrt
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/rsqrt
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-exponential
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_exponential
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-gamma
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_gamma
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-generalized-negative-binomial
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_generalized_negative_binomial
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-multinomial
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_multinomial
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-negative-binomial
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_negative_binomial
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-normal
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_normal
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-poisson
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_poisson
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sample-uniform
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sample_uniform
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 scatter-nd
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/scatter_nd
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn serialize ([ndarray] (util/coerce-return (.serialize ndarray))))

(defn
 set
 ([ndarray ndarray-or-num-or-vec-of-floats]
  (util/coerce-return
   (.set
    ndarray
    (util/coerce-param
     ndarray-or-num-or-vec-of-floats
     #{"float" "float<>" "org.apache.mxnet.NDArray"})))))

(defn
 sgd-mom-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sgd_mom_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sgd-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sgd_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn shape ([ndarray] (util/coerce-return (.shape ndarray))))

(defn
 shuffle
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/shuffle
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sigmoid
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sigmoid
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sign
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sign
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 signsgd-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/signsgd_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 signum-update
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/signum_update
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sin
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sin
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sinh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sinh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn size ([ndarray] (util/coerce-return (.size ndarray))))

(defn
 slice-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/slice_axis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 slice-like
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/slice_like
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 smooth-l1
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/smooth_l1
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softmax
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/softmax
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softmax-cross-entropy
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/softmax_cross_entropy
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 softsign
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/softsign
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sort
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sort
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 split
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/split
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sqrt
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sqrt
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 square
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/square
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 squeeze
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/squeeze
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 stack
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/stack
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 stop-gradient
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/stop_gradient
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sum
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sum
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 sum-axis
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/sum_axis
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 swapaxes
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/swapaxes
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 take
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/take
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 tan
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/tan
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 tanh
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/tanh
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 tile
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/tile
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn to-array ([ndarray] (util/coerce-return (.toArray ndarray))))

(defn to-scalar ([ndarray] (util/coerce-return (.toScalar ndarray))))

(defn
 topk
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/topk
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 tracing-enabled
 ([ndarray] (util/coerce-return (.tracingEnabled ndarray))))

(defn
 transpose
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/transpose
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 trunc
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/trunc
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn unary-- ([ndarray] (util/coerce-return (.unary_$minus ndarray))))

(defn
 uniform
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/uniform
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn
 wait-to-read
 ([ndarray] (util/coerce-return (.waitToRead ndarray))))

(defn waitall ([] (util/coerce-return (NDArray/waitall))))

(defn
 where
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/where
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

(defn writable ([ndarray] (util/coerce-return (.writable ndarray))))

(defn
 zeros-like
 ([& nd-array-and-params]
  (util/coerce-return
   (NDArray/zeros_like
    (util/coerce-param
     nd-array-and-params
     #{"scala.collection.Seq"})))))

