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

(ns infer.objectdetector-example-test
  (:require [infer.objectdetector-example :refer [detect-single-image
                                                  detect-images-in-dir]]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]))

(def model-dir "models/")
(def image-dir "images/")
(def model-path-prefix (str model-dir "resnet50_ssd/resnet50_ssd_model"))
(def image-file (str image-dir "dog.jpg"))

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/get_ssd_data.sh"))

(defn create-detector []
  (let [descriptors [{:name "data"
                      :shape [1 3 512 512]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-object-detector factory)))

(deftest test-single-detection
  (let [detector (create-detector)
        predictions (detect-single-image detector image-file)]
    (is (some? predictions))
    (is (= 5 (count predictions)))
    (is (every? #(= 2 (count %)) predictions))
    (is (every? #(string? (first %)) predictions))
    (is (every? #(= 5 (count (second %))) predictions))
    (is (every? #(< 0 (first (second %)) 1) predictions))
    (is (= ["car" "bicycle" "dog" "bicycle" "person"]
           (map first predictions)))))

(deftest test-batch-detection
  (let [detector (create-detector)
        batch-predictions (detect-images-in-dir detector image-dir)
        predictions (first batch-predictions)]
    (is (some? batch-predictions))
    (is (= 5 (count predictions)))
    (is (every? #(= 2 (count %)) predictions))
    (is (every? #(string? (first %)) predictions))
    (is (every? #(= 5 (count (second %))) predictions))
    (is (every? #(< 0 (first (second %)) 1) predictions))))
