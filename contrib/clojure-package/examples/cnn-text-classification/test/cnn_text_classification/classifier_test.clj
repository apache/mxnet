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

(ns cnn-text-classification.classifier-test
  (:require [clojure.test :refer :all]
            [org.apache.clojure-mxnet.module :as module]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.context :as context]
            [cnn-text-classification.classifier :as classifier]))

(deftest classifier-with-embeddings-test
  (let [train (classifier/train-convnet
               {:devs [(context/default-context)]
                :embedding-size 50
                :batch-size 10
                :test-size 100
                :num-epoch 1
                :max-examples 1000
                :pretrained-embedding :glove})]
    (is (= ["data"] (util/scala-vector->vec (module/data-names train))))
    (is (= 20 (count (ndarray/->vec (-> train module/outputs ffirst)))))))

(deftest classifier-without-embeddings-test
  (let [train (classifier/train-convnet
               {:devs [(context/default-context)]
                :embedding-size 50
                :batch-size 10
                :test-size 100
                :num-epoch 1
                :max-examples 1000
                :pretrained-embedding nil})]
    (is (= ["data"] (util/scala-vector->vec (module/data-names train))))
    (is (= 20 (count (ndarray/->vec (-> train module/outputs ffirst)))))))
