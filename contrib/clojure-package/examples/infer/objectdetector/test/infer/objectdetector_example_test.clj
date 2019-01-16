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
        predictions (detect-single-image detector image-file)
        {:keys [class prob x-min x-max y-min y-max] :as pred} (first predictions)]
    (is (some? predictions))
    (is (= 5 (count predictions)))
    (is (string? class))
    (is (< 0.8 prob))
    (is (every? #(< 0 % 1) [x-min x-max y-min y-max]))
    (is (= #{"dog" "person" "bicycle" "car"} (set (mapv :class predictions))))))

(deftest test-batch-detection
  (let [detector (create-detector)
        batch-predictions (detect-images-in-dir detector image-dir)
        predictions (first batch-predictions)
        {:keys [class prob x-min x-max y-min y-max] :as pred} (first predictions)]
    (is (some? batch-predictions))
    (is (= 5 (count predictions)))
    (is (string? class))
    (is (< 0.8 prob))
    (every? #(< 0 % 1) [x-min x-max y-min y-max])
    (is (= #{"dog" "person" "bicycle" "car"} (set (mapv :class predictions))))))
