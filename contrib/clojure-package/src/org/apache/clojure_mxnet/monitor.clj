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

(ns org.apache.clojure-mxnet.monitor
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet Monitor)))

(defmacro monitor
  "Monitor outputs, weights, and gradients for debugging.
  -  interval Number of batches between printing.
  -  stat-func A function that computes statistics of tensors.
                   Takes a NDArray and returns a NDArray. defaults
                   to mean absolute value |x|/size(x). Function must be in the form of clojure (fn [x])"
  [interval stat-fun]
  `(new Monitor (int ~interval) (util/scala-fn ~stat-fun)))

(defn tic
  "Start collecting stats for current batch.
   Call before forward"
  [monitor]
  (doto monitor
    (.tic)))

(defn toc
  "End collecting for current batch and return results.
   Call after computation of current batch."
  [monitor]
  (map util/tuple->vec (util/scala-vector->vec (.toVector (.toc monitor)))))
