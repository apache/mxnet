(ns org.apache.clojure-mxnet.ndarray-api
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet NDArrayAPI Shape)))

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
 ([ndarray string option]
  (util/coerce-return
   (NDArrayAPI/Activation
    (util/coerce-param ndarray #{"org.apache.mxnet.NDArray"})
    (util/coerce-param string #{"java.lang.String"})
    (util/coerce-param option #{"scala.Option"}))))
 ([ndarray string] (activation ndarray string nil)))

(defn
 batch-norm
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   option-5
   option-6
   option-7]
  (util/coerce-return
   (NDArrayAPI/BatchNorm
    (util/coerce-param ndarray #{"org.apache.mxnet.NDArray"})
    (util/coerce-param ndarray-1 #{"org.apache.mxnet.NDArray"})
    (util/coerce-param ndarray-2 #{"org.apache.mxnet.NDArray"})
    (util/coerce-param ndarray-3 #{"org.apache.mxnet.NDArray"})
    (util/coerce-param ndarray-4 #{"org.apache.mxnet.NDArray"})
    (util/coerce-param option #{"scala.Option"})
    (util/coerce-param option-1 #{"scala.Option"})
    (util/coerce-param option-2 #{"scala.Option"})
    (util/coerce-param option-3 #{"scala.Option"})
    (util/coerce-param option-4 #{"scala.Option"})
    (util/coerce-param option-5 #{"scala.Option"})
    (util/coerce-param option-6 #{"scala.Option"})
    (util/coerce-param option-7 #{"scala.Option"}))))
 ([ndarray ndarray-1 ndarray-2 ndarray-3 ndarray-4]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   nil
   nil
   nil
   nil
   nil
   nil
   nil
   nil))
 ([ndarray ndarray-1 ndarray-2 ndarray-3 ndarray-4 option]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   nil
   nil
   nil
   nil
   nil
   nil
   nil))
 ([ndarray ndarray-1 ndarray-2 ndarray-3 ndarray-4 option option-1]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   nil
   nil
   nil
   nil
   nil
   nil))
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   nil
   nil
   nil
   nil
   nil))
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   nil
   nil
   nil
   nil))
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   nil
   nil
   nil))
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   option-5]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   option-5
   nil
   nil))
 ([ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   option-5
   option-6]
  (batch-norm
   ndarray
   ndarray-1
   ndarray-2
   ndarray-3
   ndarray-4
   option
   option-1
   option-2
   option-3
   option-4
   option-5
   option-6
   nil)))

