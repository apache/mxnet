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

(ns org.apache.clojure-mxnet.callback-test
  (:require [org.apache.clojure-mxnet.callback :as callback]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(deftest test-speedometer
  (let [speedometer (callback/speedometer 1)
        metric (eval-metric/accuracy)]
    (eval-metric/update metric [(ndarray/ones [2])] [(ndarray/ones [2 3])])
    ;;; only side effects of logging
    (callback/invoke speedometer 0 1 metric)
    (callback/invoke speedometer 0 2 metric)
    (callback/invoke speedometer 0 3 metric)
    (callback/invoke speedometer 0 10 metric)
    (callback/invoke speedometer 0 50 metric)
    (callback/invoke speedometer 0 100 metric)))
