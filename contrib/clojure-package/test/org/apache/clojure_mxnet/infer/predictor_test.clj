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
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as shape]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :refer [split]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.util :as util]))

(def model-dir "data/")
(def model-path-prefix (str model-dir "resnet-18/resnet-18"))
(def width 224)
(def height 224)

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/infer/get_resnet_18_data.sh"))

(defn create-predictor []
  (let [descriptors [{:name "data"
                      :shape [1 3 height width]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-predictor factory)))

(deftest predictor-test-with-ndarray
  (let [predictor (create-predictor)
        image-ndarray (-> "test/test-images/kitten.jpg"
                           infer/load-image-from-file
                           (infer/reshape-image width height)
                           (infer/buffered-image-to-pixels [3 width height])
                           (ndarray/expand-dims 0))
        predictions (infer/predict-with-ndarray predictor [image-ndarray])
        synset-file (-> (io/file model-path-prefix)
                        (.getParent)
                        (io/file "synset.txt"))
        synset-names (split (slurp synset-file) #"\n")
        [best-index] (ndarray/->int-vec (ndarray/argmax (first predictions) 1))
        best-prediction (synset-names best-index)]
    (is (= "n02123159 tiger cat" best-prediction))))

(deftest predictor-test
  (let [predictor (create-predictor)
        image-ndarray (-> "test/test-images/kitten.jpg"
                          infer/load-image-from-file
                          (infer/reshape-image width height)
                          (infer/buffered-image-to-pixels [3 width height])
                          (ndarray/expand-dims 0))
        predictions (infer/predict predictor [(ndarray/->vec image-ndarray)])
        synset-file (-> (io/file model-path-prefix)
                        (.getParent)
                        (io/file "synset.txt"))
        synset-names (split (slurp synset-file) #"\n")
        ndarray-preds (ndarray/array (first predictions) [1 1000])
        [best-index] (ndarray/->int-vec (ndarray/argmax ndarray-preds 1))
        best-prediction (synset-names best-index)]
    (is (= "n02123159 tiger cat" best-prediction))))
