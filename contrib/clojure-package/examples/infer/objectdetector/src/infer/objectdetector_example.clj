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

(ns infer.objectdetector-example
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [clojure.java.io :as io]
            [clojure.string :refer [join]]
            [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(defn check-valid-dir
  "Check that the input directory exists"
  [input-dir]
  (let [dir (io/file input-dir)]
    (and
     (.exists dir)
     (.isDirectory dir))))

(defn check-valid-file
  "Check that the file exists"
  [input-file]
  (.exists (io/file input-file)))

(def cli-options
  [["-m" "--model-path-prefix PREFIX" "Model path prefix"
    :default "models/resnet50_ssd/resnet50_ssd_model"
    :validate [#(check-valid-file (str % "-symbol.json"))
               "Model path prefix is invalid"]]
   ["-i" "--input-image IMAGE" "Input image"
    :default "images/dog.jpg"
    :validate [check-valid-file "Input file not found"]]
   ["-d" "--input-dir IMAGE_DIR" "Input directory"
    :default "images/"
    :validate [check-valid-dir "Input directory not found"]]
   ["-h" "--help"]])

(defn print-predictions
  "Print image detector predictions for the given input file"
  [predictions width height]
  (println (apply str (repeat 80 "=")))
  (doseq [[label prob-and-bounds] predictions]
    (println (format
              "Class: %s Prob=%.5f Coords=(%.3f, %.3f, %.3f, %.3f)"
              label
              (aget prob-and-bounds 0)
              (* (aget prob-and-bounds 1) width)
              (* (aget prob-and-bounds 2) height)
              (* (aget prob-and-bounds 3) width)
              (* (aget prob-and-bounds 4) height))))
  (println (apply str (repeat 80 "="))))

(defn detect-single-image
  "Detect objects in a single image and print top-5 predictions"
  [detector input-image]
  (let [image (infer/load-image-from-file input-image)
        topk 5
        [predictions] (infer/detect-objects detector image topk)]
    predictions))

(defn detect-images-in-dir
  "Detect objects in all jpg images in the directory"
  [detector input-dir]
  (let [batch-size 20
        image-file-batches (->> input-dir
                                io/file
                                file-seq
                                (filter #(.isFile %))
                                (filter #(re-matches #".*\.jpg$" (.getPath %)))
                                (mapv #(.getPath %))
                                (partition-all batch-size))]
    (apply
     concat
     (for [image-files image-file-batches]
       (let [image-batch (infer/load-image-paths image-files)
             topk 5]
         (infer/detect-objects-batch detector image-batch topk))))))

(defn run-detector
  "Runs an image detector based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image input-dir
                device device-id]} options
        width 512 height 512
        descriptors [{:name "data"
                      :shape [1 3 height width]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)
        detector (infer/create-object-detector
                  factory
                  {:contexts [(context/default-context)]})]
    (println "Object detection on a single image")
    (print-predictions (detect-single-image detector input-image) width height)
    (println "Object detection on images in a directory")
    (doseq [predictions (detect-images-in-dir detector input-dir)]
      (print-predictions predictions width height))))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-detector options))))
