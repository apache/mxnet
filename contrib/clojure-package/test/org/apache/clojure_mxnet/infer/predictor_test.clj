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

(ns org.apache.clojure-mxnet.infer.predictor-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as shape]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]))

(def model-dir "data/")
(def model-path-prefix (str model-dir "resnet-18/resnet-18"))
(def width 224)
(def height 224)

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/infer/get_resnet_18_data.sh"))

(defn create-predictor []
  (let [descriptors [(mx-io/data-desc {:name "data"
                                       :shape [1 3 height width]
                                       :layout layout/NCHW
                                       :dtype dtype/FLOAT32})]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-predictor factory)))

(deftest predictor-test
  (let [predictor (create-predictor)
        image-ndarray (-> "test/test-images/kitten.jpg"
                          infer/load-image-from-file
                          (infer/reshape-image width height)
                          (infer/buffered-image-to-pixels
                           (shape/->shape [3 width height]))
                          (ndarray/expand-dims 0))
        [predictions] (infer/predict-with-ndarray predictor [image-ndarray])
        [best-index] (ndarray/->int-vec (ndarray/argmax predictions 1))]
    ; Should match the tiger cat index in synset file
    (is (= 282 best-index))))
