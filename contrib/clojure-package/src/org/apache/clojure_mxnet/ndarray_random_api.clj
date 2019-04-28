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

(ns org.apache.clojure-mxnet.ndarray-random-api
  "Experimental NDArray Random API"
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as mx-context]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.reflect :as r]
            [t6.from-scala.core :refer [$] :as $])
  (:import (org.apache.mxnet NDArrayAPI)))

;; loads the generated functions into the namespace
(do (clojure.core/load "gen/ndarray_random_api"))
