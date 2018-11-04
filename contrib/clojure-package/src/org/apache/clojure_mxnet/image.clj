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
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet Image)))

;; Flags for conversion of images
(def GRAYSCALE 0)
(def COLOR 1)

(defn decode-image
  "Decodes an image from an input stream"
  ([input-stream {:keys [color-flag to-rgb output]
                  :or {color-flag COLOR to-rgb true output nil}
                  :as opts}]
   (Image/imDecode input-stream color-flag to-rgb ($/option output)))
  ([input-stream]
   (decode-image input-stream {})))

(defn read-image
  "Reads an image file and returns an ndarray"
  ([filename {:keys [color-flag to-rgb output]
              :or {color-flag nil to-rgb nil output nil}
              :as opts}] 
   (Image/imRead 
    filename 
    ($/option color-flag)
    ($/option to-rgb)
    ($/option output)))
  ([filename]
   (read-image filename {})))

(defn resize-image
  "Resizes the image array to (width, height)"
  ([input w h {:keys [interpolation output]
             :or {interpolation nil output nil}
             :as opts}]
   (Image/imResize input w h ($/option interpolation) ($/option output)))
  ([input w h]
   (resize-image input w h {})))

(defn apply-border
  "Pad image border"
  ([input top bottom left right 
    {:keys [fill-type value values output]
     :or {fill-type nil value nil values nil output nil}
     :as opts}]
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
  (Image/fixedCrop input x0 y0 w h))

(defn to-image
  "Convert a NDArray image to a real image"
  [input]
  (Image/toImage input))

  
