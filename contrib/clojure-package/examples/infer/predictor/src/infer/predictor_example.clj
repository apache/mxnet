(ns infer.predictor-example
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.image :as image]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as shape]
            [clojure.java.io :as io]
            [clojure.string :refer [join split]]
            [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

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
   ["-i" "--input-image IMAGE" "Image path"
    :default "images/kitten.jpg"
    :validate [check-valid-file "Input image path not found"]]
   [nil "--device [cpu|gpu]" "Device"
    :default "cpu"
    :validate [#(#{"cpu" "gpu"} %) "Device must be one of cpu or gpu"]]
   [nil "--device-id INT" "Device ID"
    :default 0]
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
      image/read-image
      (image/resize-image width height)
      ; HWC -> CHW
      (ndarray/transpose (shape/->shape [2 0 1]))
      (ndarray/expand-dims 0)
      (ndarray/as-type dtype/FLOAT32)))

(defn do-inference
  "Run inference using given predictor"
  [predictor image]
  (let [[predictions] (infer/predict-with-ndarray predictor [image])]
    predictions))

(defn postprocess
  [model-path-prefix predictions]
  (let [synset-file (-> model-path-prefix
                        io/file
                        (.getParent)
                        (io/file "synset.txt"))
        synset-names (split (slurp synset-file) #"\n")
        [max-idx] (ndarray/->int-vec (ndarray/argmax predictions 1))]
    (print-prediction (synset-names max-idx))))

(defn run-predictor
  "Runs an image classifier based on options provided"
  [options]
  (let [{:keys [model-path-prefix input-image device device-id]} options
        ctx (if (= device "cpu")
              (context/cpu device-id)
              (context/gpu device-id))
        width 224
        height 224
        descriptors [(mx-io/data-desc {:name "data"
                                       :shape [1 3 height width]
                                       :layout layout/NCHW
                                       :dtype dtype/FLOAT32})]
        factory (infer/model-factory model-path-prefix descriptors)
        predictor (infer/create-predictor factory {:contexts [ctx]})
        image (preprocess input-image width height)
        predictions (do-inference predictor image)]
    (postprocess model-path-prefix predictions)))

(defn -main
  [& args]
  (let [{:keys [options summary errors] :as opts}
        (parse-opts args cli-options)]
    (cond
      (:help options) (println summary)
      (some? errors) (println (join "\n" errors))
      :else (run-predictor options))))
