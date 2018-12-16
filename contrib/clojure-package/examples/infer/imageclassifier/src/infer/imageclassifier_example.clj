(ns infer.imageclassifier-example
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.io :as mx-io]
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
  (let [file (io/file input-file)]
    (.exists file)))

(def cli-options
  [["-m" "--model-path-prefix PREFIX" "Model path prefix"
    :default "models/resnet-152/resnet-152"
    :validate [#(check-valid-file (str % "-symbol.json"))
               "Model path prefix is invalid"]]
   ["-i" "--input-image IMAGE" "Input image"
    :default "images/kitten.jpg"
    :validate [check-valid-file "Input file not found"]]
   ["-d" "--input-dir IMAGE_DIR" "Input directory"
    :default "images/"
    :validate [check-valid-dir "Input directory not found"]]
   [nil "--device [cpu|gpu]" "Device"
    :default "cpu"
    :validate [#(#{"cpu" "gpu"} %) "Device must be one of cpu or gpu"]]
   [nil "--device-id INT" "Device ID"
    :default 0]
   ["-h" "--help"]])

(defn print-predictions
  "Print image classifier predictions for the given input file"
  [input-file predictions]
  (println (apply str (repeat 80 "=")))
  (println "Input file:" input-file)
  (doseq [[label probability] predictions]
    (println (format "Class: %s Probability=%.8f" label probability)))
  (println (apply str (repeat 80 "="))))

(defn classify-single-image
  "Classify a single image and print top-5 predictions"
  [classifier input-image]
  (let [image (infer/load-image-from-file input-image)
        topk 5
        [predictions] (infer/classify-image classifier image topk)]
    (print-predictions input-image predictions)))

(defn classify-images-in-dir
  "Classify all jpg images in the directory"
  [classifier input-dir]
  (let [batch-size 20
        image-file-batches (->> input-dir
                                io/file
                                file-seq
                                (filter #(.isFile %))
                                (filter #(re-matches #".*\.jpg$" (.getPath %)))
                                (mapv #(.getPath %))
                                (partition-all batch-size))]
    (doseq [image-files image-file-batches]
      (let [image-batch (infer/load-image-paths image-files)
            topk 5]
        (doseq [[input-image preds]
                (map list
                     image-files
                     (infer/classify-image-batch classifier image-batch topk))]
          (print-predictions input-image preds))))))

(defn run-classifier
  "Runs an image classifier based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image input-dir
                device device-id]} options
        ctx (if (= device "cpu")
              (context/cpu device-id)
              (context/gpu device-id))
        descriptors [(mx-io/data-desc {:name "data"
                                       :shape [1 3 224 224]
                                       :layout layout/NCHW
                                       :dtype dtype/FLOAT32})]
        factory (infer/model-factory model-path-prefix descriptors)
        classifier (infer/create-image-classifier
                    factory {:contexts [ctx]})]
    (classify-single-image classifier input-image)
    (classify-images-in-dir classifier input-dir)))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-classifier options))))
