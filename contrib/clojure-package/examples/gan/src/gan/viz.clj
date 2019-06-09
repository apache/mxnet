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

(ns gan.viz
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.io :as mx-io])
  (:import (nu.pattern OpenCV)
           (org.opencv.core Core CvType Mat Size)
           (org.opencv.imgproc Imgproc)
           (org.opencv.imgcodecs Imgcodecs)))

;;; Viz stuff
(OpenCV/loadShared)

(defn clip [x]
  (->> x
       (mapv #(* 255 %))
       (mapv #(cond
                (< % 0) 0
                (> % 255) 255
                :else (int %)))
       (mapv #(.byteValue %))))

(defn get-img [raw-data channels height width flip]
  (let [totals (* height width)
        img (if (> channels 1)
              ;; rgb image
              (let [[ra ga ba] (byte-array (partition totals raw-data))
                    rr (new Mat height width (CvType/CV_8U))
                    gg (new Mat height width (CvType/CV_8U))
                    bb (new Mat height width (CvType/CV_8U))
                    result (new Mat)]
                (.put rr (int 0) (int 0) ra)
                (.put gg (int 0) (int 0) ga)
                (.put bb (int 0) (int 0) ba)
                (Core/merge (java.util.ArrayList. [bb gg rr]) result)
                result)
              ;; gray image
              (let [result (new Mat height width (CvType/CV_8U))
                    _ (.put result (int 0) (int 0) (byte-array raw-data))]
                result))]
    (do
      (if flip
        (let [result (new Mat)
              _ (Core/flip img result (int 0))]
          result)
        img))))

(defn im-sav [{:keys [title output-path x flip]
               :or {flip false} :as g-mod}]
  (let [shape (mx-shape/->vec (ndarray/shape x))
        _ (assert (== 4 (count shape)))
        [n c h w] shape
        totals (* h w)
        raw-data (byte-array (clip (ndarray/to-array x)))
        row (.intValue (Math/sqrt n))
        col row
        line-arrs (into [] (partition (* col c totals) raw-data))
        line-mats (mapv (fn [line]
                          (let [img-arr (into [] (partition (* c totals) line))
                                col-mats (new Mat)
                                src (mapv (fn [arr] (get-img (into [] arr) c h w flip)) img-arr)
                                _ (Core/hconcat (java.util.ArrayList. src) col-mats)]
                            col-mats))
                        line-arrs)
        result (new Mat)
        resized-img (new Mat)
        _ (Core/vconcat (java.util.ArrayList. line-mats) result)]
    (do
      (Imgproc/resize result resized-img (new Size (* (.width result) 1.5) (* (.height result) 1.5)))
      (Imgcodecs/imwrite (str output-path title ".jpg") resized-img)
      (Thread/sleep 1000))))
