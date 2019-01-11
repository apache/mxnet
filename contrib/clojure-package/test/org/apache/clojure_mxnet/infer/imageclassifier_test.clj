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

(ns org.apache.clojure-mxnet.infer.imageclassifier-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]))

(def model-dir "data/")
(def model-path-prefix (str model-dir "resnet-18/resnet-18"))

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/infer/get_resnet_18_data.sh"))

(defn create-classifier []
  (let [descriptors [{:name "data"
                      :shape [1 3 224 224]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-image-classifier factory)))

(deftest test-single-classification
  (let [classifier (create-classifier)
        image (infer/load-image-from-file "test/test-images/kitten.jpg")
        [predictions-all] (infer/classify-image classifier image)
        [predictions-with-default-dtype] (infer/classify-image classifier image 10)
        [predictions] (infer/classify-image classifier image 5 dtype/FLOAT32)]
    (is (= 1000 (count predictions-all)))
    (is (= 10 (count predictions-with-default-dtype)))
    (is (some? predictions))
    (is (= 5 (count predictions)))
    (is (every? #(= 2 (count %)) predictions))
    (is (every? #(string? (first %)) predictions))
    (is (every? #(float? (second %)) predictions))
    (is (every? #(< 0 (second %) 1) predictions))
    (is (= ["n02123159 tiger cat"
            "n02124075 Egyptian cat"
            "n02123045 tabby, tabby cat"
            "n02127052 lynx, catamount"
            "n02128757 snow leopard, ounce, Panthera uncia"]
           (map first predictions)))))

(deftest test-batch-classification
  (let [classifier (create-classifier)
        image-batch (infer/load-image-paths ["test/test-images/kitten.jpg"
                                             "test/test-images/Pug-Cookie.jpg"])
        batch-predictions-all (infer/classify-image-batch classifier image-batch)
        batch-predictions-with-default-dtype (infer/classify-image-batch classifier image-batch 10)
        batch-predictions (infer/classify-image-batch classifier image-batch 5 dtype/FLOAT32)
        predictions (first batch-predictions)]
    (is (= 1000 (count (first batch-predictions-all))))
    (is (= 10 (count (first batch-predictions-with-default-dtype))))
    (is (some? batch-predictions))
    (is (= 5 (count predictions)))
    (is (every? #(= 2 (count %)) predictions))
    (is (every? #(string? (first %)) predictions))
    (is (every? #(float? (second %)) predictions))
    (is (every? #(< 0 (second %) 1) predictions))))
