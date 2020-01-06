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
            [clojure.test :refer [deftest is use-fixtures run-tests]]
            [test-helper])
  (:import (javax.imageio ImageIO)
           (java.io File)))


(test-helper/load-test-images)

(def tmp-dir (System/getProperty "java.io.tmpdir"))
(def image-path (.getAbsolutePath (io/file tmp-dir "Pug-Cookie.jpg")))
(def image-src-path "test/test-images/Pug-Cookie.jpg")

(defn- cp
  "Copy from filepath `from` to filepath `to`."
  [from to]
  (with-open [in (io/input-stream (io/file from))
              out (io/output-stream (io/file to))]
    (io/copy in out)))

(defn- rm
  "Removes `filepath`."
  [filepath]
  (io/delete-file filepath))

(defn- with-file
  "Provides `src-path` in `dest-path` for the test function `f` to use."
  [src-path dest-path]
  (fn [f]
    (cp src-path dest-path)
    (f)
    (rm dest-path)))

(use-fixtures :once (with-file image-src-path image-path))

(deftest test-decode-image
  (let [img-arr (image/decode-image (io/input-stream image-path))
        img-arr-2 (image/decode-image (io/input-stream image-path)
                                      {:color-flag image/GRAYSCALE})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-decode
  (let [img-arr (image/decode (io/input-stream image-path))
        img-arr-2 (image/decode (io/input-stream image-path)
                                {:color :grayscale})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-read-image
  (let [img-arr (image/read-image image-path)
        img-arr-2 (image/read-image image-path {:color-flag image/GRAYSCALE})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-read
  (let [img-arr (image/read image-path)
        img-arr-2 (image/read image-path {:color :grayscale})]
    (is (= [576 1024 3] (ndarray/shape-vec img-arr)))
    (is (= [576 1024 1] (ndarray/shape-vec img-arr-2)))))

(deftest test-resize-image
  (let [img-arr (image/read image-path)
        resized-arr (image/resize-image img-arr 224 224)]
    (is (= [224 224 3] (ndarray/shape-vec resized-arr)))))

(deftest test-resize
  (let [img-arr (image/read image-path)
        resized-arr (image/resize img-arr 224 224)]
    (is (= [224 224 3] (ndarray/shape-vec resized-arr)))))

(deftest test-fixed-crop
  (let [img-arr (image/read image-path)
        cropped-arr (image/fixed-crop img-arr 0 0 224 224)]
    (is (= [224 224 3] (ndarray/shape-vec cropped-arr)))))

(deftest test-apply-border
  (let [img-arr (image/read image-path)
        padded-arr (image/apply-border img-arr 1 1 1 1)]
    (is (= [578 1026 3] (ndarray/shape-vec padded-arr)))))

(deftest test-to-image
  (let [img-arr (image/read image-path)
        resized-arr (image/resize img-arr 224 224)
        new-img (image/to-image resized-arr)]
    (is (ImageIO/write new-img "png" (io/file tmp-dir "out.png")))))

(deftest test-ndarray->image
  (let [img-arr (image/read image-path)
        resized-arr (image/resize img-arr 224 224)
        new-img (image/ndarray->image resized-arr)]
    (is (ImageIO/write new-img "png" (io/file tmp-dir "out.png")))))

(deftest test-draw-bounding-box!
  (let [orig-img (ImageIO/read (new File image-path))
        new-img  (image/draw-bounding-box!
                   orig-img
                   [{:x-min 190 :x-max 850 :y-min 50 :y-max 450}
                    {:x-min 200 :x-max 350 :y-min 440 :y-max 530}]
                   {:stroke 2
                    :names ["pug" "cookie"]
                    :transparency 0.8
                    :font-size-mult 2.0})]
    (is (ImageIO/write new-img "png" (io/file tmp-dir "out.png")))))
