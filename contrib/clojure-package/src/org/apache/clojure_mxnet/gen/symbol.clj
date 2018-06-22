(ns org.apache.clojure-mxnet.symbol
    (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max
                              min repeat reverse set sort take to-array empty sin
                              get apply shuffle])
    (:require [org.apache.clojure-mxnet.util :as util])
    (:import (org.apache.mxnet Symbol)))

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
 ([sym sym-or-object]
  (util/coerce-return
   (.$div
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 div$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$div$mDc$sp sym (util/nil-or-coerce-param double #{"double"})))))

(defn
 div$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$div$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 div$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$div$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 >
 ([sym sym-or-object]
  (util/coerce-return
   (.$greater
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 >=
 ([sym sym-or-object]
  (util/coerce-return
   (.$greater$eq
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 >=$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$greater$eq$mDc$sp
    sym
    (util/nil-or-coerce-param double #{"double"})))))

(defn
 >=$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$greater$eq$mFc$sp
    sym
    (util/nil-or-coerce-param num #{"float"})))))

(defn
 >=$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$greater$eq$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 >$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$greater$mDc$sp
    sym
    (util/nil-or-coerce-param double #{"double"})))))

(defn
 >$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$greater$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 >$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$greater$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 <
 ([sym sym-or-object]
  (util/coerce-return
   (.$less
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 <=
 ([sym sym-or-object]
  (util/coerce-return
   (.$less$eq
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 <=$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$less$eq$mDc$sp
    sym
    (util/nil-or-coerce-param double #{"double"})))))

(defn
 <=$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$less$eq$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 <=$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$less$eq$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 <$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$less$mDc$sp sym (util/nil-or-coerce-param double #{"double"})))))

(defn
 <$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$less$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 <$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$less$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 -
 ([sym object-or-sym]
  (util/coerce-return
   (.$minus
    sym
    (util/nil-or-coerce-param
     object-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 -$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$minus$mDc$sp sym (util/nil-or-coerce-param double #{"double"})))))

(defn
 -$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$minus$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 -$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$minus$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 %
 ([sym sym-or-object]
  (util/coerce-return
   (.$percent
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 %$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$percent$mDc$sp
    sym
    (util/nil-or-coerce-param double #{"double"})))))

(defn
 %$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$percent$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 %$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$percent$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 +
 ([sym object-or-sym]
  (util/coerce-return
   (.$plus
    sym
    (util/nil-or-coerce-param
     object-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 +$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$plus$mDc$sp sym (util/nil-or-coerce-param double #{"double"})))))

(defn
 +$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$plus$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 +$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$plus$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 *
 ([sym sym-or-object]
  (util/coerce-return
   (.$times
    sym
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 *$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$times$mDc$sp sym (util/nil-or-coerce-param double #{"double"})))))

(defn
 *$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$times$mFc$sp sym (util/nil-or-coerce-param num #{"float"})))))

(defn
 *$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$times$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 **
 ([sym object-or-sym]
  (util/coerce-return
   (.$times$times
    sym
    (util/nil-or-coerce-param
     object-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 **$m-dc$sp
 ([sym double]
  (util/coerce-return
   (.$times$times$mDc$sp
    sym
    (util/nil-or-coerce-param double #{"double"})))))

(defn
 **$m-fc$sp
 ([sym num]
  (util/coerce-return
   (.$times$times$mFc$sp
    sym
    (util/nil-or-coerce-param num #{"float"})))))

(defn
 **$m-ic$sp
 ([sym num]
  (util/coerce-return
   (.$times$times$mIc$sp sym (util/nil-or-coerce-param num #{"int"})))))

(defn
 activation
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Activation
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (activation
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (activation
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (activation
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 batch-norm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/BatchNorm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (batch-norm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (batch-norm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (batch-norm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 batch-norm-v1
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/BatchNorm_v1
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (batch-norm-v1
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (batch-norm-v1
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (batch-norm-v1
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 bilinear-sampler
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/BilinearSampler
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (bilinear-sampler
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (bilinear-sampler
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (bilinear-sampler
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 block-grad
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/BlockGrad
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (block-grad
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (block-grad
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (block-grad
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cast
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Cast
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cast
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cast
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cast
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 concat
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Concat
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (concat
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (concat
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (concat
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 convolution
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Convolution
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (convolution
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (convolution
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (convolution
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 convolution-v1
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Convolution_v1
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (convolution-v1
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (convolution-v1
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (convolution-v1
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 correlation
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Correlation
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (correlation
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (correlation
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (correlation
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 crop
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Crop
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (crop
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (crop
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (crop
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 custom
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Custom
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (custom
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (custom
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (custom
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 deconvolution
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Deconvolution
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (deconvolution
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (deconvolution
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (deconvolution
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 dropout
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Dropout
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (dropout
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (dropout
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (dropout
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 element-wise-sum
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ElementWiseSum
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (element-wise-sum
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (element-wise-sum
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (element-wise-sum
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 embedding
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Embedding
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (embedding
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (embedding
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (embedding
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 flatten
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Flatten
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (flatten
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (flatten
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (flatten
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 fully-connected
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/FullyConnected
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (fully-connected
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (fully-connected
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (fully-connected
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 grid-generator
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/GridGenerator
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (grid-generator
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (grid-generator
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (grid-generator
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 group
 ([symbol-list]
  (util/coerce-return
   (Symbol/Group
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})))))

(defn
 identity-attach-kl-sparse-reg
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/IdentityAttachKLSparseReg
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (identity-attach-kl-sparse-reg
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (identity-attach-kl-sparse-reg
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (identity-attach-kl-sparse-reg
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 instance-norm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/InstanceNorm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (instance-norm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (instance-norm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (instance-norm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 l2-normalization
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/L2Normalization
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (l2-normalization
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (l2-normalization
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (l2-normalization
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 lrn
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/LRN
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (lrn
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (lrn
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (lrn
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 layer-norm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/LayerNorm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (layer-norm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (layer-norm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (layer-norm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 leaky-re-lu
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/LeakyReLU
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (leaky-re-lu
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (leaky-re-lu
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (leaky-re-lu
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linear-regression-output
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/LinearRegressionOutput
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linear-regression-output
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linear-regression-output
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linear-regression-output
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 logistic-regression-output
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/LogisticRegressionOutput
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (logistic-regression-output
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (logistic-regression-output
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (logistic-regression-output
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 mae-regression-output
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/MAERegressionOutput
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (mae-regression-output
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (mae-regression-output
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (mae-regression-output
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 make-loss
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/MakeLoss
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (make-loss
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (make-loss
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (make-loss
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 pad
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Pad
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (pad
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (pad
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (pad
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 pooling
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Pooling
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (pooling
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (pooling
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (pooling
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 pooling-v1
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Pooling_v1
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (pooling-v1
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (pooling-v1
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (pooling-v1
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rnn
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/RNN
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rnn
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rnn
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rnn
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 roi-pooling
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ROIPooling
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (roi-pooling
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (roi-pooling
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (roi-pooling
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 reshape
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Reshape
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (reshape
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (reshape
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (reshape
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 svm-output
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SVMOutput
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (svm-output
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (svm-output
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (svm-output
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sequence-last
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SequenceLast
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sequence-last
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sequence-last
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sequence-last
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sequence-mask
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SequenceMask
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sequence-mask
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sequence-mask
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sequence-mask
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sequence-reverse
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SequenceReverse
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sequence-reverse
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sequence-reverse
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sequence-reverse
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 slice-channel
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SliceChannel
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (slice-channel
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (slice-channel
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (slice-channel
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softmax
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/Softmax
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softmax
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softmax
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softmax
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softmax-activation
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SoftmaxActivation
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softmax-activation
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softmax-activation
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softmax-activation
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softmax-output
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SoftmaxOutput
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softmax-output
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softmax-output
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softmax-output
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 spatial-transformer
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SpatialTransformer
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (spatial-transformer
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (spatial-transformer
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (spatial-transformer
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 swap-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/SwapAxis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (swap-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (swap-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (swap-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 up-sampling
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/UpSampling
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (up-sampling
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (up-sampling
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (up-sampling
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 abs
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/abs
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (abs
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (abs
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (abs
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 adam-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/adam_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (adam-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (adam-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (adam-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 add-n
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/add_n
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (add-n
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (add-n
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (add-n
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 apply
 ([sym sym-name kwargs-map]
  (util/coerce-return
   (.apply
    sym
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})))))

(defn
 arccos
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arccos
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arccos
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arccos
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arccos
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 arccosh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arccosh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arccosh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arccosh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arccosh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 arcsin
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arcsin
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arcsin
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arcsin
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arcsin
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 arcsinh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arcsinh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arcsinh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arcsinh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arcsinh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 arctan
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arctan
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arctan
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arctan
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arctan
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 arctanh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/arctanh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (arctanh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (arctanh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (arctanh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 argmax
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/argmax
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (argmax
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (argmax
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (argmax
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 argmax-channel
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/argmax_channel
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (argmax-channel
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (argmax-channel
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (argmax-channel
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 argmin
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/argmin
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (argmin
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (argmin
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (argmin
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 argsort
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/argsort
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (argsort
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (argsort
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (argsort
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 attr
 ([sym sym-name]
  (util/coerce-return
   (.attr
    sym
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})))))

(defn attr-map ([sym] (util/coerce-return (.attrMap sym))))

(defn
 batch-dot
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/batch_dot
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (batch-dot
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (batch-dot
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (batch-dot
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 batch-take
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/batch_take
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (batch-take
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (batch-take
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (batch-take
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-add
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_add
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-add
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-add
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-add
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-axes
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_axes
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-axes
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-axes
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-axes
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_axis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-div
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_div
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-div
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-div
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-div
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-equal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_equal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-equal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-equal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-equal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-greater
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_greater
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-greater
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-greater
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-greater
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-greater-equal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_greater_equal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-greater-equal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-greater-equal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-greater-equal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-hypot
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_hypot
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-hypot
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-hypot
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-hypot
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-lesser
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_lesser
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-lesser
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-lesser
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-lesser
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-lesser-equal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_lesser_equal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-lesser-equal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-lesser-equal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-lesser-equal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-maximum
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_maximum
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-maximum
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-maximum
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-maximum
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-minimum
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_minimum
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-minimum
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-minimum
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-minimum
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-minus
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_minus
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-minus
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-minus
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-minus
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-mod
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_mod
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-mod
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-mod
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-mod
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-mul
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_mul
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-mul
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-mul
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-mul
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-not-equal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_not_equal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-not-equal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-not-equal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-not-equal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-plus
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_plus
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-plus
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-plus
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-plus
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-power
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_power
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-power
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-power
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-power
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-sub
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_sub
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-sub
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-sub
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-sub
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 broadcast-to
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/broadcast_to
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (broadcast-to
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (broadcast-to
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (broadcast-to
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cast
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/cast
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cast
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cast
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cast
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cast-storage
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/cast_storage
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cast-storage
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cast-storage
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cast-storage
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cbrt
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/cbrt
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cbrt
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cbrt
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cbrt
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 ceil
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ceil
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (ceil
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (ceil
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (ceil
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 choose-element-0index
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/choose_element_0index
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (choose-element-0index
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (choose-element-0index
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (choose-element-0index
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 clip
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/clip
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (clip
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (clip
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (clip
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn clone ([sym] (util/coerce-return (.clone sym))))

(defn
 concat
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/concat
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (concat
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (concat
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (concat
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cos
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/cos
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cos
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cos
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cos
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 cosh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/cosh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (cosh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (cosh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (cosh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 create-from-listed-symbols
 ([sym-name sym-name-1 kwargs-map Symbol<> kwargs-map-1]
  (util/coerce-return
   (Symbol/createFromListedSymbols
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param sym-name-1 #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param Symbol<> #{"org.apache.mxnet.Symbol<>"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"})))))

(defn
 create-from-listed-symbols-no-check
 ([sym-name sym-name-1 kwargs-map Symbol<> kwargs-map-1]
  (util/coerce-return
   (Symbol/createFromListedSymbolsNoCheck
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param sym-name-1 #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param Symbol<> #{"org.apache.mxnet.Symbol<>"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"})))))

(defn
 create-from-named-symbols
 ([sym-name sym-name-1 kwargs-map kwargs-map-1 kwargs-map-1]
  (util/coerce-return
   (Symbol/createFromNamedSymbols
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param sym-name-1 #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"})))))

(defn
 create-from-named-symbols-no-check
 ([sym-name sym-name-1 kwargs-map kwargs-map-1]
  (util/coerce-return
   (Symbol/createFromNamedSymbolsNoCheck
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param sym-name-1 #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"})))))

(defn creation-trace ([sym] (util/coerce-return (.creationTrace sym))))

(defn
 crop
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/crop
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (crop
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (crop
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (crop
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn debug-str ([sym] (util/coerce-return (.debugStr sym))))

(defn
 degrees
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/degrees
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (degrees
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (degrees
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (degrees
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn dispose ([sym] (util/coerce-return (.dispose sym))))

(defn
 dot
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/dot
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (dot
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (dot
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (dot
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 elemwise-add
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/elemwise_add
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (elemwise-add
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (elemwise-add
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (elemwise-add
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 elemwise-div
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/elemwise_div
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (elemwise-div
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (elemwise-div
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (elemwise-div
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 elemwise-mul
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/elemwise_mul
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (elemwise-mul
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (elemwise-mul
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (elemwise-mul
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 elemwise-sub
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/elemwise_sub
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (elemwise-sub
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (elemwise-sub
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (elemwise-sub
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 equal
 ([sym-or-sym-or-object object-or-sym-or-sym]
  (util/coerce-return
   (Symbol/equal
    (util/nil-or-coerce-param
     sym-or-sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})
    (util/nil-or-coerce-param
     object-or-sym-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 exp
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/exp
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (exp
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (exp
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (exp
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 expand-dims
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/expand_dims
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (expand-dims
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (expand-dims
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (expand-dims
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 expm1
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/expm1
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (expm1
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (expm1
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (expm1
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 fill-element-0index
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/fill_element_0index
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (fill-element-0index
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (fill-element-0index
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (fill-element-0index
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn finalize ([sym] (util/coerce-return (.finalize sym))))

(defn
 fix
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/fix
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (fix
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (fix
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (fix
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 flatten
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/flatten
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (flatten
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (flatten
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (flatten
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 flip
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/flip
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (flip
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (flip
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (flip
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 floor
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/floor
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (floor
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (floor
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (floor
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 ftml-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ftml_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (ftml-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (ftml-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (ftml-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 ftrl-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ftrl_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (ftrl-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (ftrl-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (ftrl-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 gamma
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/gamma
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (gamma
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (gamma
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (gamma
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 gammaln
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/gammaln
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (gammaln
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (gammaln
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (gammaln
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 gather-nd
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/gather_nd
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (gather-nd
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (gather-nd
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (gather-nd
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 get
 ([sym sym-name-or-num]
  (util/coerce-return
   (.get
    sym
    (util/nil-or-coerce-param
     sym-name-or-num
     #{"int" "java.lang.String"})))))

(defn get-internals ([sym] (util/coerce-return (.getInternals sym))))

(defn
 greater
 ([sym-or-sym sym-or-object]
  (util/coerce-return
   (Symbol/greater
    (util/nil-or-coerce-param sym-or-sym #{"org.apache.mxnet.Symbol"})
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 greater-equal
 ([sym-or-sym sym-or-object]
  (util/coerce-return
   (Symbol/greaterEqual
    (util/nil-or-coerce-param sym-or-sym #{"org.apache.mxnet.Symbol"})
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn handle ([sym] (util/coerce-return (.handle sym))))

(defn
 identity
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/identity
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (identity
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (identity
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (identity
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 infer-shape
 ([sym kwargs-map-or-symbol-list]
  (util/coerce-return
   (.inferShape
    sym
    (util/nil-or-coerce-param
     kwargs-map-or-symbol-list
     #{"scala.collection.Seq" "scala.collection.immutable.Map"}))))
 ([sym vec-or-strings vec-of-ints vec-of-ints-1]
  (util/coerce-return
   (.inferShape
    sym
    (util/nil-or-coerce-param vec-or-strings #{"java.lang.String<>"})
    (util/nil-or-coerce-param vec-of-ints #{"int<>"})
    (util/nil-or-coerce-param vec-of-ints-1 #{"int<>"})))))

(defn
 infer-type
 ([sym symbol-list-or-kwargs-map]
  (util/coerce-return
   (.inferType
    sym
    (util/nil-or-coerce-param
     symbol-list-or-kwargs-map
     #{"scala.collection.Seq" "scala.collection.immutable.Map"})))))

(defn is-disposed ([sym] (util/coerce-return (.isDisposed sym))))

(defn
 khatri-rao
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/khatri_rao
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (khatri-rao
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (khatri-rao
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (khatri-rao
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 lesser
 ([sym-or-sym sym-or-object]
  (util/coerce-return
   (Symbol/lesser
    (util/nil-or-coerce-param sym-or-sym #{"org.apache.mxnet.Symbol"})
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 lesser-equal
 ([sym-or-sym sym-or-object]
  (util/coerce-return
   (Symbol/lesserEqual
    (util/nil-or-coerce-param sym-or-sym #{"org.apache.mxnet.Symbol"})
    (util/nil-or-coerce-param
     sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 linalg-gelqf
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_gelqf
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-gelqf
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-gelqf
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-gelqf
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-gemm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_gemm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-gemm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-gemm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-gemm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-gemm2
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_gemm2
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-gemm2
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-gemm2
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-gemm2
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-potrf
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_potrf
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-potrf
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-potrf
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-potrf
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-potri
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_potri
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-potri
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-potri
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-potri
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-sumlogdiag
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_sumlogdiag
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-sumlogdiag
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-sumlogdiag
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-sumlogdiag
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-syrk
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_syrk
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-syrk
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-syrk
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-syrk
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-trmm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_trmm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-trmm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-trmm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-trmm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 linalg-trsm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/linalg_trsm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (linalg-trsm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (linalg-trsm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (linalg-trsm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn list-arguments ([sym] (util/coerce-return (.listArguments sym))))

(defn list-attr ([sym] (util/coerce-return (.listAttr sym))))

(defn
 list-auxiliary-states
 ([sym] (util/coerce-return (.listAuxiliaryStates sym))))

(defn list-outputs ([sym] (util/coerce-return (.listOutputs sym))))

(defn
 load
 ([sym-name]
  (util/coerce-return
   (Symbol/load
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})))))

(defn
 load-json
 ([sym-name]
  (util/coerce-return
   (Symbol/loadJson
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})))))

(defn
 log
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/log
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (log
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (log
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (log
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 log10
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/log10
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (log10
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (log10
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (log10
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 log1p
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/log1p
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (log1p
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (log1p
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (log1p
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 log2
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/log2
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (log2
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (log2
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (log2
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 log-dispose-warning
 ([sym] (util/coerce-return (.logDisposeWarning sym))))

(defn
 log-softmax
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/log_softmax
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (log-softmax
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (log-softmax
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (log-softmax
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 make-loss
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/make_loss
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (make-loss
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (make-loss
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (make-loss
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 max-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/max_axis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (max-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (max-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (max-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 mean
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/mean
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (mean
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (mean
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (mean
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 min-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/min_axis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (min-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (min-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (min-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 mp-sgd-mom-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/mp_sgd_mom_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (mp-sgd-mom-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (mp-sgd-mom-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (mp-sgd-mom-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 mp-sgd-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/mp_sgd_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (mp-sgd-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (mp-sgd-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (mp-sgd-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 nanprod
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/nanprod
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (nanprod
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (nanprod
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (nanprod
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 nansum
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/nansum
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (nansum
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (nansum
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (nansum
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 negative
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/negative
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (negative
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (negative
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (negative
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 norm
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/norm
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (norm
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (norm
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (norm
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 normal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/normal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (normal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (normal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (normal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 not-equal
 ([sym-or-sym-or-object sym-or-object-or-sym]
  (util/coerce-return
   (Symbol/notEqual
    (util/nil-or-coerce-param
     sym-or-sym-or-object
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})
    (util/nil-or-coerce-param
     sym-or-object-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 one-hot
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/one_hot
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (one-hot
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (one-hot
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (one-hot
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 ones-like
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/ones_like
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (ones-like
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (ones-like
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (ones-like
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 org.apache.mxnet.symbol
 ([sym long]
  (util/coerce-return
   (.org.apache.mxnet.Symbol
    sym
    (util/nil-or-coerce-param long #{"long"})))))

(defn
 pad
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/pad
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (pad
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (pad
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (pad
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 pick
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/pick
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (pick
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (pick
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (pick
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 pow
 ([sym-or-object-or-sym object-or-sym-or-sym]
  (util/coerce-return
   (Symbol/pow
    (util/nil-or-coerce-param
     sym-or-object-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})
    (util/nil-or-coerce-param
     object-or-sym-or-sym
     #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))

(defn
 prod
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/prod
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (prod
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (prod
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (prod
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 radians
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/radians
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (radians
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (radians
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (radians
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-exponential
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_exponential
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-exponential
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-exponential
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-exponential
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-gamma
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_gamma
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-gamma
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-gamma
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-gamma
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-generalized-negative-binomial
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_generalized_negative_binomial
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-generalized-negative-binomial
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-generalized-negative-binomial
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-generalized-negative-binomial
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-negative-binomial
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_negative_binomial
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-negative-binomial
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-negative-binomial
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-negative-binomial
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-normal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_normal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-normal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-normal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-normal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-poisson
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_poisson
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-poisson
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-poisson
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-poisson
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 random-uniform
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/random_uniform
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (random-uniform
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (random-uniform
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (random-uniform
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rcbrt
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/rcbrt
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rcbrt
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rcbrt
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rcbrt
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 reciprocal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/reciprocal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (reciprocal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (reciprocal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (reciprocal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 relu
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/relu
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (relu
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (relu
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (relu
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 repeat
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/repeat
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (repeat
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (repeat
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (repeat
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 reshape
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/reshape
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (reshape
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (reshape
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (reshape
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 reshape-like
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/reshape_like
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (reshape-like
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (reshape-like
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (reshape-like
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 reverse
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/reverse
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (reverse
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (reverse
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (reverse
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rint
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/rint
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rint
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rint
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rint
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rmsprop-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/rmsprop_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rmsprop-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rmsprop-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rmsprop-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rmspropalex-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/rmspropalex_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rmspropalex-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rmspropalex-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rmspropalex-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 round
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/round
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (round
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (round
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (round
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 rsqrt
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/rsqrt
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (rsqrt
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (rsqrt
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (rsqrt
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-exponential
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_exponential
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-exponential
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-exponential
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-exponential
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-gamma
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_gamma
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-gamma
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-gamma
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-gamma
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-generalized-negative-binomial
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_generalized_negative_binomial
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-generalized-negative-binomial
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-generalized-negative-binomial
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-generalized-negative-binomial
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-multinomial
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_multinomial
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-multinomial
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-multinomial
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-multinomial
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-negative-binomial
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_negative_binomial
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-negative-binomial
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-negative-binomial
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-negative-binomial
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-normal
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_normal
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-normal
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-normal
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-normal
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-poisson
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_poisson
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-poisson
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-poisson
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-poisson
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sample-uniform
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sample_uniform
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sample-uniform
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sample-uniform
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sample-uniform
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 save
 ([sym sym-name]
  (util/coerce-return
   (.save
    sym
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})))))

(defn
 scatter-nd
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/scatter_nd
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (scatter-nd
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (scatter-nd
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (scatter-nd
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sgd-mom-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sgd_mom_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sgd-mom-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sgd-mom-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sgd-mom-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sgd-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sgd_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sgd-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sgd-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sgd-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 shuffle
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/shuffle
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (shuffle
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (shuffle
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (shuffle
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sigmoid
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sigmoid
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sigmoid
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sigmoid
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sigmoid
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sign
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sign
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sign
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sign
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sign
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 signsgd-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/signsgd_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (signsgd-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (signsgd-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (signsgd-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 signum-update
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/signum_update
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (signum-update
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (signum-update
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (signum-update
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sin
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sin
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sin
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sin
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sin
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sinh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sinh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sinh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sinh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sinh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 slice
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/slice
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (slice
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (slice
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (slice
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 slice-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/slice_axis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (slice-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (slice-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (slice-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 slice-like
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/slice_like
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (slice-like
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (slice-like
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (slice-like
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 smooth-l1
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/smooth_l1
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (smooth-l1
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (smooth-l1
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (smooth-l1
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softmax
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/softmax
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softmax
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softmax
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softmax
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softmax-cross-entropy
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/softmax_cross_entropy
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softmax-cross-entropy
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softmax-cross-entropy
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softmax-cross-entropy
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 softsign
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/softsign
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (softsign
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (softsign
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (softsign
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sort
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sort
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sort
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sort
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sort
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 split
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/split
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (split
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (split
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (split
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sqrt
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sqrt
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sqrt
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sqrt
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sqrt
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 square
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/square
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (square
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (square
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (square
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 squeeze
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/squeeze
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (squeeze
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (squeeze
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (squeeze
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 stack
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/stack
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (stack
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (stack
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (stack
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 stop-gradient
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/stop_gradient
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (stop-gradient
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (stop-gradient
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (stop-gradient
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sum
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sum
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sum
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sum
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sum
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 sum-axis
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/sum_axis
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (sum-axis
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (sum-axis
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (sum-axis
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 swapaxes
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/swapaxes
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (swapaxes
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (swapaxes
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (swapaxes
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 take
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/take
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (take
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (take
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (take
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 tan
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/tan
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (tan
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (tan
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (tan
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 tanh
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/tanh
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (tanh
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (tanh
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (tanh
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 tile
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/tile
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (tile
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (tile
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (tile
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn to-json ([sym] (util/coerce-return (.toJson sym))))

(defn
 topk
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/topk
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (topk
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (topk
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (topk
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 tracing-enabled
 ([sym] (util/coerce-return (.tracingEnabled sym))))

(defn
 transpose
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/transpose
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (transpose
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (transpose
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (transpose
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 trunc
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/trunc
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (trunc
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (trunc
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (trunc
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 uniform
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/uniform
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (uniform
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (uniform
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (uniform
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 where
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/where
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (where
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (where
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (where
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

(defn
 zeros-like
 ([sym-name kwargs-map symbol-list kwargs-map-1]
  (util/coerce-return
   (Symbol/zeros_like
    (util/nil-or-coerce-param sym-name #{"java.lang.String"})
    (util/nil-or-coerce-param
     kwargs-map
     #{"scala.collection.immutable.Map"})
    (util/nil-or-coerce-param symbol-list #{"scala.collection.Seq"})
    (util/nil-or-coerce-param
     kwargs-map-1
     #{"scala.collection.immutable.Map"}))))
 ([sym-name attr-map kwargs-map]
  (zeros-like
   sym-name
   (util/convert-symbol-map attr-map)
   (util/empty-list)
   (util/convert-symbol-map kwargs-map)))
 ([sym-name kwargs-map-or-vec-or-sym]
  (zeros-like
   sym-name
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil)))
 ([kwargs-map-or-vec-or-sym]
  (zeros-like
   nil
   nil
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (util/empty-list)
    (util/coerce-param
     kwargs-map-or-vec-or-sym
     #{"scala.collection.Seq"}))
   (if
    (clojure.core/map? kwargs-map-or-vec-or-sym)
    (org.apache.clojure-mxnet.util/convert-symbol-map
     kwargs-map-or-vec-or-sym)
    nil))))

