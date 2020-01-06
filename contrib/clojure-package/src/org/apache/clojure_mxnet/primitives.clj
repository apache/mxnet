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

(ns org.apache.clojure-mxnet.primitives
  (:import (org.apache.mxnet MX_PRIMITIVES$MX_FLOAT MX_PRIMITIVES$MX_Double
                             MX_PRIMITIVES$MX_PRIMITIVE_TYPE)))


;;; Defines customer mx primitives that can be used for mathematical computations
;;; in NDArrays to control precision. Currently Float and Double are supported

;;; For purposes of automatic conversion in ndarray functions, doubles are default
;; to specify using floats you must use a Float

(defn mx-float
  "Creates a MXNet float primitive"
  [num]
  (new MX_PRIMITIVES$MX_FLOAT num))

(defn mx-double
  "Creates a MXNet double primitive"
  [num]
  (new MX_PRIMITIVES$MX_Double num))

(defn ->num
  "Returns the underlying number value"
  [primitive]
  (.data primitive))

(defn primitive? [x]
  (instance? MX_PRIMITIVES$MX_PRIMITIVE_TYPE x))

