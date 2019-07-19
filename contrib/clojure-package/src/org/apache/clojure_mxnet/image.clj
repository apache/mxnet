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
  "Image API of Clojure package."
  (:refer-clojure :exclude [read])
  (:require [t6.from-scala.core :refer [$ $$] :as $]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet Image NDArray)
           (java.awt.image BufferedImage)
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

(defn ^:deprecated decode-image
  "DEPRECATED: use `decode` instead.

   Decodes an image from an input stream with OpenCV
    `input-stream`: `InputStream` - Contains the binary encoded image
    `color-flag`: 0 or 1 - Convert decoded image to grayscale (0) or color (1)
    `to-rgb`: boolean - Whether to convert decoded image to mxnet's default RGB
            format (instead of opencv's default BGR)
    `output`: nil or `NDArray`
    returns: `NDArray` with dtype uint8

  Ex:
    (decode-image input-stream)
    (decode-image input-stream {:color-flag 1})
    (decode-image input-stream {:color-flag 0 :output nd})"
  ([input-stream {:keys [color-flag to-rgb output]
                  :or {color-flag COLOR to-rgb true output nil}
                  :as opts}]
   (util/validate! ::input-stream input-stream "Invalid input stream")
   (util/validate! ::decode-image-opts opts "Invalid options for decoding")
   (Image/imDecode input-stream color-flag to-rgb ($/option output)))
  ([input-stream]
   (decode-image input-stream {})))

(s/def ::color #{:grayscale :color})
(s/def ::decode-image-opts-2 (s/keys :opt-un [::color ::to-rgb ::output]))

(defn- color->int [color]
  (case color
    :grayscale 0
    :color 1))

(defn decode
  "Decodes an image from an input stream with OpenCV.
    `input-stream`: `InputStream` - Contains the binary encoded image
    `color`: keyword in `#{:color :grayscale}` - Convert decoded image to
             grayscale or color
    `to-rgb`: boolean - Whether to convert decoded image to mxnet's default RGB
            format (instead of opencv's default BGR)
    `output`: nil or `NDArray`
    returns: `NDArray` with dtype uint8

  Ex:
    (decode input-stream)
    (decode input-stream {:color :color})
    (decode input-stream {:color :grayscale :output nd})"
  ([input-stream {:keys [color to-rgb output]
                  :or {color :color to-rgb true output nil}
                  :as opts}]
   (util/validate! ::input-stream input-stream "Invalid input stream")
   (util/validate! ::decode-image-opts-2 opts "Invalid options for decoding")
   (Image/imDecode input-stream (color->int color) to-rgb ($/option output)))
  ([input-stream]
   (decode input-stream {})))

(s/def ::filename string?)
(s/def ::optional-color-flag
  (s/or :none nil? :some ::color-flag))
(s/def ::optional-to-rgb
  (s/or :none nil? :some ::to-rgb))

(defn ^:deprecated read-image
  "DEPRECATED: use `read` instead.

   Reads an image file and returns an ndarray with OpenCV. It returns image in
   RGB by default instead of OpenCV's default BGR.
    `filename`: string - Name of the image file to be loaded
    `color-flag`: 0 or 1 - Convert decoded image to grayscale (0) or color (1)
    `to-rgb`: boolean - Whether to convert decoded image to mxnet's default RGB
            format (instead of opencv's default BGR)
    `output`: nil or `NDArray`
    returns: `NDArray` with dtype uint8

   Ex:
     (read-image \"cat.jpg\")
     (read-image \"cat.jpg\" {:color-flag 0})
     (read-image \"cat.jpg\" {:color-flag 1 :output nd})"
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

(defn read
  "Reads an image file and returns an ndarray with OpenCV. It returns image in
   RGB by default instead of OpenCV's default BGR.
    `filename`: string - Name of the image file to be loaded
    `color`: keyword in `#{:color :grayscale}` - Convert decoded image to
             grayscale or color
    `to-rgb`: boolean - Whether to convert decoded image to mxnet's default RGB
            format (instead of opencv's default BGR)
    `output`: nil or `NDArray`
    returns: `NDArray` with dtype uint8

   Ex:
     (read \"cat.jpg\")
     (read \"cat.jpg\" {:color :grayscale})
     (read \"cat.jpg\" {:color :color :output nd})"
  ([filename {:keys [color to-rgb output]
              :or {color :color to-rgb nil output nil}
              :as opts}]
   (util/validate! ::filename filename "Invalid filename")
   (util/validate! ::color color "Invalid color")
   (util/validate! ::optional-to-rgb to-rgb "Invalid conversion flag")
   (util/validate! ::output output "Invalid output")
   (Image/imRead
    filename
    ($/option (when color (color->int color)))
    ($/option to-rgb)
    ($/option output)))
  ([filename]
   (read filename {})))

(s/def ::int int?)
(s/def ::optional-int (s/or :none nil? :some int?))

(defn ^:deprecated resize-image
  "DEPRECATED: use `resize` instead.

   Resizes the image array to (width, height)
   `input`: `NDArray` - source image in NDArray
   `w`: int - Width of resized image
   `h`: int - Height of resized image
   `interpolation`: Interpolation method. Default is INTER_LINEAR
   `ouput`: nil or `NDArray`
   returns: `NDArray`

   Ex:
     (resize-image nd-img 300 300)
     (resize-image nd-img 28 28 {:output nd})"
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

(defn resize
  "Resizes the image array to (width, height)
   `input`: `NDArray` - source image in NDArray
   `w`: int - Width of resized image
   `h`: int - Height of resized image
   `interpolation`: Interpolation method. Default is INTER_LINEAR
   `ouput`: nil or `NDArray`
   returns: `NDArray`

   Ex:
     (resize nd-img 300 300)
     (resize nd-img 28 28 {:output nd})"
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
   (resize input w h {})))

(defn apply-border
  "Pad image border with OpenCV.
   `input`: `NDArray` - source image in NDArray
   `top`: int - Top margin
   `bottom`: int - Bottom margin
   `left`: int - Left margin
   `right`: int - Right margin
   `fill-type`: nil or Filling type - Default BORDER_CONSTANT
   `value`: nil or double - Deprecated, use `values` instead
   `values`: Fill with value(RGB or gray), up to 4 channels
   `output`: nil or `NDArray`
   returns: `NDArray`

   Ex:
     (apply-border img-nd 1 1 1 1)
     (apply-border img-nd 3 3 0 0)"
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
  "Return a fixed crop of the image.
   `input`: `NDArray` - Source image in NDArray
   `x0`: int - Starting x point
   `y0`: int - Starting y point
   `w`: int - Width of the image
   `h`: int - Height of the image
   returns: cropped `NDArray`

   Ex:
     (fixed-crop nd-img 0 0 28 28)
     (fixed-crop nd-img 10 0 100 300)"
  [input x0 y0 w h]
  (util/validate! ::ndarray input "Invalid input array")
  (util/validate! ::int x0 "Invalid starting x coordinate")
  (util/validate! ::int y0 "Invalid starting y coordinate")
  (util/validate! ::int w "Invalid width")
  (util/validate! ::int h "Invalid height")
  (Image/fixedCrop input x0 y0 w h))

(defn rgb-array?
  "Returns whether the ndarray is in the RGB format
   `input`: `NDArray` - Source image in NDArray
   returns: boolean"
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

(defn ^:deprecated to-image
  "DEPRECATED: user `ndarray->image` instead.

   Convert a NDArray image in RGB format to a real image.
   `input`: `NDArray` - Source image in NDArray
   returns: `BufferedImage`"
  [input]
  (util/validate! ::to-image-ndarray input "Invalid input array")
  (Image/toImage input))

(defn ndarray->image
  "Convert a NDArray image in RGB format to a real image.
   `input`: `NDArray` - Source image in NDArray
   returns: `BufferedImage`"
  [input]
  (util/validate! ::to-image-ndarray input "Invalid input array")
  (Image/toImage input))

(s/def ::buffered-image #(instance? BufferedImage %))
(s/def ::x-min number?)
(s/def ::x-max number?)
(s/def ::y-min number?)
(s/def ::y-max number?)
(s/def ::coordinate (s/keys :req-un [::x-min ::x-max ::y-min ::y-max]))
(s/def ::coordinates (s/coll-of ::coordinate))
(s/def ::names (s/nilable (s/coll-of string?)))
(s/def ::stroke (s/and integer? pos?))
(s/def ::font-size-mult (s/and float? pos?))
(s/def ::transparency (s/and float? #(<= 0.0 % 1.0)))
(s/def ::coordinates-names
  (fn [[coordinates names]] (= (count coordinates) (count names))))

(defn- convert-coordinate
  "Convert bounding box coordinate to Scala correct types."
  [{:keys [x-min x-max y-min y-max]}]
  {:xmin (int x-min)
   :xmax (int x-max)
   :ymin (int y-min)
   :ymax (int y-max)})

(defn draw-bounding-box!
  "Draw bounding boxes on `buffered-image` and Mutate the input image.
  `buffered-image`: BufferedImage
  `coordinates`: collection of {:xmin int :xmax int :ymin int :ymax int}
  `font-size-mult`: positive float - Font size multiplier
  `names`: collection of strings - List of names for the bounding boxes
  `stroke`: positive integer - thickness of the bounding box
  `transparency`: float in (0.0, 1.0) - Transparency of the bounding box
  returns: Modified `buffered-image`
  Ex:
    (draw-bounding-box! img [{:x-min 0 :x-max 100 :y-min 0 :y-max 100}])
    (draw-bounding-box! [{:x-min 190 :x-max 850 :y-min 50 :y-max 450}
                         {:x-min 200 :x-max 350 :y-min 440 :y-max 530}]
                        {:stroke 2
                         :names [\"pug\" \"cookie\"]
                         :transparency 0.8
                         :font-size-mult 2.0})"
  ([buffered-image coordinates]
   (draw-bounding-box! buffered-image coordinates {}))
  ([buffered-image coordinates
    {:keys [names stroke font-size-mult transparency]
     :or {stroke 3 font-size-mult 1.0 transparency 1.0}
     :as opts}]
  (util/validate! ::buffered-image buffered-image "Invalid input image")
  (util/validate! ::coordinates coordinates "Invalid input coordinates")
  (util/validate! ::names names "Invalid input names")
  (util/validate! ::stroke stroke "Invalid input stroke")
  (util/validate! ::font-size-mult font-size-mult "Invalid input font-size-mult")
  (util/validate! ::transparency transparency "Invalid input transparency")
  (when (pos? (count names))
    (util/validate!  ::coordinates-names [coordinates names] "Invalid number of names"))
  (Image/drawBoundingBox
    buffered-image
    (->> coordinates
         (map convert-coordinate)
         (map util/convert-map)
         (into-array))
    (util/->option (into-array names))
    (util/->option (int stroke))
    (util/->option (float font-size-mult))
    (util/->option (float transparency)))
  buffered-image))
