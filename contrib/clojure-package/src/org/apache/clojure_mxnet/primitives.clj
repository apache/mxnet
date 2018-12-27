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

