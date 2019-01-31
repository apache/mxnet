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

(ns infer.predictor-example-test
  (:require [infer.predictor-example :refer [preprocess
                                             do-inference
                                             postprocess]]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]))

(def model-dir "models/")
(def image-file "images/kitten.jpg")
(def model-path-prefix (str model-dir "resnet-18/resnet-18"))
(def width 224)
(def height 224)

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/get_resnet_18_data.sh"))

(defn create-predictor []
  (let [descriptors [{:name "data"
                      :shape [1 3 height width]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-predictor factory)))

(deftest predictor-test
  (let [predictor (create-predictor)
        image-ndarray (preprocess image-file width height)
        predictions (do-inference predictor image-ndarray)
        best-prediction (postprocess model-path-prefix predictions)]
    (is (= "n02123159 tiger cat" best-prediction))))
