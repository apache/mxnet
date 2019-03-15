(ns org.apache.clojure-mxnet.ndarray
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
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
 ([ndarray ndarray-or-double-or-float]
  (util/coerce-return
   (.$div
    ndarray
    (util/coerce-param
     ndarray-or-double-or-float
     #{"org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE"
       "org.apache.mxnet.NDArray"})))))

