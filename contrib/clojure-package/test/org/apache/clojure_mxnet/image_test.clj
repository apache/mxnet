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

(ns org.apache.clojure-mxnet.image-test
  (:require [org.apache.clojure-mxnet.image :as image]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [clojure.java.io :as io]
            [clojure.test :refer :all])
  (:import (javax.imageio ImageIO)))

(def tmp-dir (System/getProperty "java.io.tmpdir"))
(def image-path (.getAbsolutePath (io/file tmp-dir "Pug-Cookie.jpg")))

(defn download-image []
  (with-open [in (io/input-stream "https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg")
              out (io/output-stream (io/file image-path))]
    (io/copy in out)))

(defn delete-image []
  (io/delete-file image-path))

(defn with-downloaded-image [f]
  (download-image)
  (f)
  (delete-image))

(use-fixtures :once with-downloaded-image)

(deftest test-decode-image
  (let [img-arr (image/decode-image 
                 (io/input-stream image-path))
        img-arr-2 (image/decode-image 
                   (io/input-stream image-path)
                   {:color-flag image/GRAYSCALE})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-read-image
  (let [img-arr (image/read-image image-path)
        img-arr-2 (image/read-image
                   image-path
                   {:color-flag image/GRAYSCALE})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-resize-image
  (let [img-arr (image/read-image image-path)
        resized-arr (image/resize-image img-arr 224 224)]
    (is (= [224 224 3] (ndarray/shape-vec resized-arr)))))

(deftest test-crop-image
  (let [img-arr (image/read-image image-path)
        cropped-arr (image/fixed-crop img-arr 0 0 224 224)]
    (is (= [224 224 3] (ndarray/shape-vec cropped-arr)))))

(deftest test-apply-border
  (let [img-arr (image/read-image image-path)
        padded-arr (image/apply-border img-arr 1 1 1 1)]
    (is (= [578 1026 3] (ndarray/shape-vec padded-arr)))))

(deftest test-to-image
  (let [img-arr (image/read-image image-path)
        resized-arr (image/resize-image img-arr 224 224)
        new-img (image/to-image resized-arr)]
    (is (= true (ImageIO/write new-img "png" (io/file tmp-dir "out.png"))))))
