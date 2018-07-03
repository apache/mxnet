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

(ns visualization.core
  (:require [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]))

(defn get-symbol []
  (as-> (sym/variable "data") data

    #_(sym/convolution "conv1" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    #_(sym/batch-norm "bn1" {:data data})
    #_(sym/activation "relu1" {:data data :act-type "relu"})
    #_(sym/pooling "mp1" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]}) #_(sym/convolution "conv2" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    #_(sym/batch-norm "bn2" {:data data})
    #_(sym/activation "relu2" {:data data :act-type "relu"})
    #_(sym/pooling "mp2" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]})

    (sym/flatten "fl" {:data data})
    #_(sym/fully-connected "fc2" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))

(defn test-viz []
  (let [dot (viz/plot-network (get-symbol)
                              {"data" [1 1 28 28]}
                              {:title "foo" :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot "testviz" "./")))

(defn -main [& args]
  (do (test-viz)
      (println "Check for the testviz.pdf file in the project directory")))

