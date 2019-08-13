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

(ns pre-trained-models.predict-image
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.symbol :as sym]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

;; based on https://mxnet.incubator.apache.org/tutorials/python/predict_image.html

;; run download-reset-152.sh to get the model params and json

(def model-dir "model")
(def num-channels 3)
(def h 224)
(def w 224)

(defn download [uri file]
  (with-open [in (io/input-stream uri)
              out (io/output-stream file)]
    (io/copy in out)))

(defn get-image [url show?]
  (-> url
      (cvu/mat-from-url)
      (cv/resize! (cv/new-size h w))
      (#(do (if show? (cvu/imshow %)) %))
      (cv/convert-to! cv/CV_8SC3 0.5) 
      (cvu/mat->flat-rgb-array)
      (ndarray/array [1 num-channels h w])))

(defn predict [img-url show?]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/resnet-152") :epoch 0})
        labels (-> (slurp (str model-dir "/synset.txt"))
                   (string/split #"\n"))
        nd-img (get-image img-url show?)
        prob (-> mod
                 (m/bind {:for-training false :data-shapes [{:name "data" :shape [1 num-channels h w]}]})
                 (m/forward {:data [nd-img]})
                 (m/outputs)
                 (ffirst))
        prob-with-labels (mapv (fn [p l] {:prob p :label l})
                               (ndarray/->vec prob)
                               labels)]
    (->> (sort-by :prob prob-with-labels)
         (reverse)
         (take 5))))

(defn feature-extraction []
  (let [nd-img (get-image "http://animalsbirds.com/wp-content/uploads/2016/07/Animal-Cat-HD-Wallpapers.jpg" false)
        mod (-> (m/load-checkpoint {:prefix (str model-dir "/resnet-152") :epoch 0})
                (m/bind {:for-training false :data-shapes [{:name "data" :shape [1 num-channels h w]}]}))
        fe-sym (-> (m/symbol mod)
                   (sym/get-internals)
                   (sym/get "flatten0_output"))
        fe-mod (-> (m/module fe-sym {:label-names nil})
                   (m/bind {:for-training false :data-shapes [{:name "data" :shape [1 num-channels h w]}]})
                   (m/init-params {:arg-params (m/arg-params mod) :aux-params (m/aux-params mod)}))]
    (-> fe-mod
        (m/forward {:data [nd-img]})
        (m/outputs)
        (ffirst)
        (ndarray/shape)
        (mx-shape/->vec))))

(defn -main [& args]
  (println 
   (predict 
    (or (first args)
        "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" )
        true)))

(comment

  (predict "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" true)
  ;; ({:prob 0.69066674, :label "n02122948 kitten, kitty"}
  ;;  {:prob 0.04466057, :label "n01323155 kit"}
  ;;  {:prob 0.029682875, :label "n01318894 pet"}
  ;;  {:prob 0.028944906, :label "n02122878 tabby, queen"}
  ;;  {:prob 0.027530408, :label "n01322221 baby"})

  (predict "http://thenotoriouspug.com/wp-content/uploads/2015/01/Pug-Cookie-1920x1080-1024x576.jpg" true)
  ;; ({:prob 0.44412872, :label "n02110958 pug, pug-dog"}
  ;;  {:prob 0.093773685,
  ;;   :label "n13905792 wrinkle, furrow, crease, crinkle, seam, line"}
  ;;  {:prob 0.02395489, :label "n01318894 pet"}
  ;;  {:prob 0.023736171,
  ;;   :label "n02084732 pooch, doggie, doggy, barker, bow-wow"}
  ;;  {:prob 0.023329297, :label "n02083346 canine, canid"})

  (feature-extraction) ;=> [1 2048]
)

