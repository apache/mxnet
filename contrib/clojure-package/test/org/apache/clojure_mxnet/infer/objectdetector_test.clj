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

(ns org.apache.clojure-mxnet.infer.objectdetector-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(def model-dir "data/")
(def model-path-prefix (str model-dir "resnet50_ssd/resnet50_ssd_model"))

(when-not (.exists (io/file (str model-path-prefix "-symbol.json")))
  (sh "./scripts/infer/get_ssd_data.sh"))

(defn create-detector []
  (let [descriptors [{:name "data"
                      :shape [1 3 512 512]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)]
    (infer/create-object-detector factory)))

(deftest test-single-detection
  (let [detector (create-detector)
        image (infer/load-image-from-file "test/test-images/kitten.jpg")
        [predictions-all] (infer/detect-objects detector image)
        [predictions] (infer/detect-objects detector image 5)
        {:keys [class prob x-min x-max y-min y-max] :as pred} (first predictions)]
    (is (some? predictions))
    (is (= 5 (count predictions)))
    (is (= 13 (count predictions-all)))
    (is (= "cat" class))
    (is (< 0.8 prob))
    (every? #(< 0 % 1) [x-min x-max y-min y-max])))

(deftest test-batch-detection
  (let [detector (create-detector)
        image-batch (infer/load-image-paths ["test/test-images/kitten.jpg"
                                             "test/test-images/Pug-Cookie.jpg"])
        [batch-predictions-all] (infer/detect-objects-batch detector image-batch)
        [predictions] (infer/detect-objects-batch detector image-batch 5)
        {:keys [class prob x-min x-max y-min y-max] :as pred} (first predictions)]
    (is (some? predictions))
    (is (= 13 (count batch-predictions-all)))
    (is (= 5 (count predictions)))
    (is (= "cat" class))
    (is (< 0.8 prob))
    (every? #(< 0 % 1) [x-min x-max y-min y-max])))

(deftest test-detection-with-ndarrays
  (let [detector (create-detector)
        image (-> (infer/load-image-from-file "test/test-images/kitten.jpg")
                  (infer/reshape-image 512 512)
                  (infer/buffered-image-to-pixels [3 512 512] dtype/FLOAT32)
                  (ndarray/expand-dims 0))
        [predictions-all] (infer/detect-objects-with-ndarrays detector [image])
        [predictions] (infer/detect-objects-with-ndarrays detector [image] 1)
        {:keys [class prob x-min x-max y-min y-max] :as pred} (first predictions)]
        (is (some? predictions-all))
        (is (= 1 (count predictions)))
        (is (= "cat" class))
        (is (< 0.8 prob))
        (every? #(< 0 % 1) [x-min x-max y-min y-max])))

