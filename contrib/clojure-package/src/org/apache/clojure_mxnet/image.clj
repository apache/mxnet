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

(ns org.apache.clojure-mxnet.image
  (:require [t6.from-scala.core :refer [$ $$] :as $]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet Image NDArray)
           (java.io InputStream)))

;; Flags for conversion of images
(def GRAYSCALE 0)
(def COLOR 1)

(s/def ::input-stream #(instance? InputStream %))
(s/def ::color-flag #{GRAYSCALE COLOR})
(s/def ::to-rgb boolean?)
(s/def ::ndarray #(instance? NDArray %))
(s/def ::output (s/or :empty nil? :ndarray ::ndarray))
(s/def ::decode-image-opts
  (s/keys :opt-un [::color-flag ::to-rgb ::output]))

(defn decode-image
  "Decodes an image from an input stream"
  ([input-stream {:keys [color-flag to-rgb output]
                  :or {color-flag COLOR to-rgb true output nil}
                  :as opts}]
   (util/validate! ::input-stream input-stream "Invalid input stream")
   (util/validate! ::decode-image-opts opts "Invalid options for decoding")
   (Image/imDecode input-stream color-flag to-rgb ($/option output)))
  ([input-stream]
   (decode-image input-stream {})))

(s/def ::filename string?)
(s/def ::optional-color-flag
  (s/or :none nil? :some ::color-flag))
(s/def ::optional-to-rgb
  (s/or :none nil? :some ::to-rgb))

(defn read-image
  "Reads an image file and returns an ndarray"
  ([filename {:keys [color-flag to-rgb output]
              :or {color-flag nil to-rgb nil output nil}
              :as opts}]
   (util/validate! ::filename filename "Invalid filename")
   (util/validate! ::optional-color-flag color-flag "Invalid color flag")
   (util/validate! ::optional-to-rgb to-rgb "Invalid conversion flag")
   (util/validate! ::output output "Invalid output")
   (Image/imRead 
    filename 
    ($/option color-flag)
    ($/option to-rgb)
    ($/option output)))
  ([filename]
   (read-image filename {})))

(s/def ::int int?)
(s/def ::optional-int (s/or :none nil? :some int?))

(defn resize-image
  "Resizes the image array to (width, height)"
  ([input w h {:keys [interpolation output]
               :or {interpolation nil output nil}
               :as opts}]
   (util/validate! ::ndarray input "Invalid input array")
   (util/validate! ::int w "Invalid width")
   (util/validate! ::int h "Invalid height")
   (util/validate! ::optional-int interpolation "Invalid interpolation")
   (util/validate! ::output output "Invalid output")
   (Image/imResize input w h ($/option interpolation) ($/option output)))
  ([input w h]
   (resize-image input w h {})))

(defn apply-border
  "Pad image border"
  ([input top bottom left right 
    {:keys [fill-type value values output]
     :or {fill-type nil value nil values nil output nil}
     :as opts}]
   (util/validate! ::ndarray input "Invalid input array")
   (util/validate! ::int top "Invalid top margin")
   (util/validate! ::int bottom "Invalid bottom margin")
   (util/validate! ::int left "Invalid left margin")
   (util/validate! ::int right "Invalid right margin")
   (util/validate! ::optional-int fill-type "Invalid fill type")
   (util/validate! ::output output "Invalid output")
   (Image/copyMakeBorder input top bottom left right
                         ($/option fill-type)
                         ($/option value)
                         ($/option values)
                         ($/option output)))
  ([input top bottom left right]
   (apply-border input top bottom left right {})))

(defn fixed-crop
  "Return a fixed crop of the image"
  [input x0 y0 w h]
  (util/validate! ::ndarray input "Invalid input array")
  (util/validate! ::int x0 "Invalid starting x coordinate")
  (util/validate! ::int y0 "Invalid starting y coordinate")
  (util/validate! ::int w "Invalid width")
  (util/validate! ::int h "Invalid height")
  (Image/fixedCrop input x0 y0 w h))

(defn rgb-array?
  "Returns whether the ndarray is in the RGB format"
  [input]
  (util/validate! ::ndarray input "Invalid input array")
  (let [shape (ndarray/shape-vec input)]
    (and
     (= 3 (count shape))
     (= 3 (shape 2)))))

(s/def ::all-bytes #(= dtype/UINT8 (ndarray/dtype %)))
(s/def ::rgb-array rgb-array?)
(s/def ::to-image-ndarray
  (s/and ::ndarray ::all-bytes ::rgb-array))

(defn to-image
  "Convert a NDArray image in RGB format to a real image"
  [input]
  (util/validate! ::to-image-ndarray input "Invalid input array")
  (Image/toImage input))
