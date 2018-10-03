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

(ns org.apache.clojure-mxnet.visualization-test
  (:require [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]
            [clojure.test :refer :all])
  (:import (org.apache.mxnet Visualization$Dot)))

(deftest test-plot-network
  (let [to-plot-sym (as-> (sym/variable "data") data
                      (sym/flatten "fl" {:data data})
                      (sym/softmax-output "softmax" {:data data}))
        dot (viz/plot-network to-plot-sym
                              {"data" [1 1 28 28]}
                              {:title "foo"
                               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (is (instance? Visualization$Dot dot))))
