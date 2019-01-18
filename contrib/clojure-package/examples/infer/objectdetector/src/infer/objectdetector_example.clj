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
            [infer.draw :as draw]
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
   ["-o" "--output-dir IMAGE_DIR" "Output directory. Defaults to results"
    :default "results/"
    :validate [check-valid-dir "Output directory not found"]]
   ["-d" "--input-dir IMAGE_DIR" "Input directory"
    :default "images/"
    :validate [check-valid-dir "Input directory not found"]]
   ["-h" "--help"]])

(defn result->map [{:keys [class prob x-min y-min x-max y-max]}]
  (hash-map
   :label class
   :confidence (int (* 100 prob))
   :top-left [x-min y-min]
   :bottom-right [x-max y-max]))

(defn print-results [results]
  (doseq [_r results]
    (println (format "Class: %s Confidence=%s Coords=(%s, %s)"
                     (_r :label)
                     (_r :confidence)
                     (_r :top-left)
                     (_r :bottom-right)))))

(defn process-results [images results output-dir]
  (dotimes [i (count images)]
    (let [image (nth images i) _results (map result->map (nth results i))]
      (println "processing: " image)
      (print-results _results)
      (draw/draw-bounds image _results output-dir))))

(defn detect-single-image
  "Detect objects in a single image and print top-5 predictions"
  [detector input-image output-dir]
  (let [image (infer/load-image-from-file input-image)
        topk 5]
    (process-results
     [input-image]
     (infer/detect-objects detector image topk)
     output-dir)))

(defn detect-images-in-dir
  "Detect objects in all jpg images in the directory"
  [detector input-dir output-dir]
  (let [batch-size 20
        image-file-batches (->> input-dir
                                io/file
                                file-seq
                                (filter #(.isFile %))
                                (filter #(re-matches #".*\.jpg$" (.getPath %)))
                                (mapv #(.getPath %))
                                (partition-all batch-size))]
    (doall
     (for [image-files image-file-batches]
       (let [image-batch (infer/load-image-paths image-files) topk 5]
         (process-results
          image-files
          (infer/detect-objects-batch detector image-batch topk)
          output-dir))))))

(defn run-detector
  "Runs an image detector based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image input-dir output-dir
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
    (println "Output results to:" output-dir ":" (.mkdir (io/file output-dir)))
    (println "Object detection on a single image")
    (detect-single-image detector input-image output-dir)
    (println "Object detection on images in a directory")
    (detect-images-in-dir detector input-dir output-dir)))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-detector options))))
