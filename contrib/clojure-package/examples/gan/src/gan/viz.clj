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
            [org.apache.clojure-mxnet.io :as mx-io]
            [opencv4.utils :as cvu]
            [opencv4.core :as cv :refer [CV_8UC1 new-matofbyte flip! imwrite new-size hconcat! vconcat! new-mat merge!]]))

(defn clip [x]
  (->> x
       (mapv #(* 255 %))
       (mapv #(cond
                (< % 0) 0
                (> % 255) 255
                :else (int %)))
       (mapv #(.byteValue %))))

(defn get-img [raw-data channels height width flip]
  (let [img (if (> channels 1)
              (throw (Exception. "Image with 3 channels (RGB) not supported"))
              ;; gray image
              (cv/>> (new-mat height width CV_8UC1) (byte-array raw-data)))]
    (if flip (flip! img 0) img)))

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
                                src (mapv (fn [arr] (get-img (into [] arr) c h w flip)) img-arr)]
                            (hconcat! src)))
                        line-arrs)]
    (-> line-mats
        (vconcat!)
        (cvu/resize-by 1.5)
        (imwrite (str output-path title ".jpg")))))