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

(ns infer.imageclassifier-example
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
    :default "models/resnet-18/resnet-18"
    :validate [#(check-valid-file (str % "-symbol.json"))
               "Model path prefix is invalid"]]
   ["-i" "--input-image IMAGE" "Input image"
    :default "images/kitten.jpg"
    :validate [check-valid-file "Input file not found"]]
   ["-d" "--input-dir IMAGE_DIR" "Input directory"
    :default "images/"
    :validate [check-valid-dir "Input directory not found"]]
   ["-h" "--help"]])

(defn print-predictions
  "Print image classifier predictions for the given input file"
  [predictions]
  (println (apply str (repeat 80 "=")))
  (doseq [p predictions]
    (println p))
  (println (apply str (repeat 80 "="))))

(defn classify-single-image
  "Classify a single image and print top-5 predictions"
  [classifier input-image]
  (let [image (infer/load-image-from-file input-image)
        topk 5
        predictions (infer/classify-image classifier image topk)]
    [predictions]))

(defn classify-images-in-dir
  "Classify all jpg images in the directory"
  [classifier input-dir]
  (let [batch-size 20
        image-file-batches (->> input-dir
                                io/file
                                file-seq
                                sort
                                reverse
                                (filter #(.isFile %))
                                (filter #(re-matches #".*\.jpg$" (.getPath %)))
                                (mapv #(.getPath %))
                                (partition-all batch-size))]
    (apply concat (for [image-files image-file-batches]
                    (let [image-batch (infer/load-image-paths image-files)
                          topk 5]
                      (infer/classify-image-batch classifier image-batch topk))))))

(defn run-classifier
  "Runs an image classifier based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image input-dir]} options
        descriptors [{:name "data"
                      :shape [1 3 224 224]
                      :layout layout/NCHW
                      :dtype dtype/FLOAT32}]
        factory (infer/model-factory model-path-prefix descriptors)
        classifier (infer/create-image-classifier
                    factory {:contexts [(context/default-context)]})]
    (println "Classifying a single image")
    (print-predictions (classify-single-image classifier input-image))
    (println "\n")
    (println "Classifying images in a directory")
    (doseq [predictions (classify-images-in-dir classifier input-dir)]
      (print-predictions predictions))))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-classifier options))))
