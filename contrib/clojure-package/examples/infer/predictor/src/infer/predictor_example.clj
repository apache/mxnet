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

(ns infer.predictor-example
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.image :as image]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [clojure.java.io :as io]
            [clojure.string :refer [join split]]
            [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(defn check-valid-file
  "Check that the file exists"
  [input-file]
  (.exists (io/file input-file)))

(def cli-options
  [["-m" "--model-path-prefix PREFIX" "Model path prefix"
    :default "models/resnet-18/resnet-18"
    :validate [#(check-valid-file (str % "-symbol.json"))
               "Model path prefix is invalid"]]
   ["-i" "--input-image IMAGE" "Image path"
    :default "images/kitten.jpg"
    :validate [check-valid-file "Input image path not found"]]
   ["-h" "--help"]])

(defn print-prediction
  [prediction]
  (println (apply str (repeat 80 "=")))
  (println prediction)
  (println (apply str (repeat 80 "="))))

(defn preprocess
  "Preprocesses image to make it ready for prediction"
  [image-path width height]
  (-> image-path
      infer/load-image-from-file
      (infer/reshape-image width height)
      (infer/buffered-image-to-pixels [3 width height])
      (ndarray/expand-dims 0)))

(defn do-inference
  "Run inference using given predictor"
  [predictor image]
  (let [predictions (infer/predict-with-ndarray predictor [image])]
    (first predictions)))

(defn postprocess
  [model-path-prefix predictions]
  (let [synset-file (-> model-path-prefix
                        io/file
                        (.getParent)
                        (io/file "synset.txt"))
        synset-names (split (slurp synset-file) #"\n")
        [max-idx] (ndarray/->int-vec (ndarray/argmax predictions 1))]
    (synset-names max-idx)))

(defn run-predictor
  "Runs an image classifier based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image]} options
        width 224
        height 224
        descriptors [{:name "data"
                      :shape [1 3 height width]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)
        predictor (infer/create-predictor
                   factory
                   {:contexts [(context/default-context)]})
        image-ndarray (preprocess input-image width height)
        predictions (do-inference predictor image-ndarray)
        best-prediction (postprocess model-path-prefix predictions)]
    (print-prediction best-prediction)))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-predictor options))))
