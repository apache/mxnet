(ns org.apache.clojure-mxnet.ndarray-sample-api
  "Experimental NDArray Random API"
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as mx-context]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.reflect :as r]
            [t6.from-scala.core :refer [$] :as $])
  (:import (org.apache.mxnet NDArrayAPI)))

;; loads the generated functions into the namespace
(do (clojure.core/load "gen/ndarray_sample_api"))
