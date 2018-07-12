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

(ns pre-trained-models.fine-tune
  (:require [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym])
  (:gen-class))

;;; From the finetune example https://mxnet.incubator.apache.org/faq/finetune.html

;; run download-resnet-50.sh to get the model params and json
;; and download-caltech.sh to get the pregenerated rec files

(def model-dir "model")
(def batch-size 16)

;;; image set is http://www.vision.caltech.edu/Image_Datasets/Caltech101/
;; Pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images

(def train-iter (mx-io/image-record-iter
                 {:path-imgrec "caltech-256/caltech-256-60-train.rec"
                  :data-name "data"
                  :label-name "softmax_label"
                  :batch-size batch-size
                  :data-shape [3 224 224]
                  :shuffle true
                  :rand-crop true
                  :rand-mirror true}))

(def val-iter (mx-io/image-record-iter
               {:path-imgrec "caltech-256/caltech-256-60-val.rec"
                :data-name "data"
                :label-name "softmax_label"
                :batch-size batch-size
                :data-shape [3 224 224]
                :rand-crop false
                :rand-mirror false}))

(defn get-model []
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/resnet-50") :epoch 0})]
    {:msymbol (m/symbol mod)
     :arg-params (m/arg-params mod)
     :aux-params (m/aux-params mod)}))

(defn get-fine-tune-model
  "msymbol: the pretrained network symbol
    arg-params: the argument parameters of the pretrained model
    num-classes: the number of classes for the fine-tune datasets
    layer-name: the layer name before the last fully-connected layer"
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args   (->> arg-params
                      (remove (fn [[k v]] (string/includes? k "fc1")))
                      (into {}))}))

(defn fit [devs msymbol arg-params aux-params]
  (let [mod (-> (m/module msymbol {:contexts devs})
                (m/bind {:data-shapes (mx-io/provide-data train-iter) :label-shapes (mx-io/provide-label val-iter)})
                (m/init-params {:arg-params arg-params :aux-params aux-params
                                :allow-missing true}))]
    (m/fit mod
           {:train-data train-iter
            :eval-data val-iter
            :num-epoch 1
            :fit-params (m/fit-params {:intializer (init/xavier {:rand-type "gaussian"
                                                                 :factor-type "in"
                                                                 :magnitude 2})
                                       :batch-end-callback (callback/speedometer batch-size 10)})})))

(defn fine-tune! [devs]
  (let [{:keys [msymbol arg-params aux-params] :as model} (get-model)
        {:keys [net new-args]} (get-fine-tune-model (merge model {:num-classes 256}))]
    (fit devs net new-args arg-params)))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Running with context devices of" devs)
    (fine-tune! devs)))

(comment

  (fine-tune! [(context/cpu)])

;INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [10]	Speed: 3.61 samples/sec	Train-accuracy=0.000000
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [20]	Speed: 3.49 samples/sec	Train-accuracy=0.005952
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [30]	Speed: 3.58 samples/sec	Train-accuracy=0.012097
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [40]	Speed: 3.49 samples/sec	Train-accuracy=0.013720
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [50]	Speed: 3.51 samples/sec	Train-accuracy=0.017157
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [60]	Speed: 3.56 samples/sec	Train-accuracy=0.017418
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [70]	Speed: 3.56 samples/sec	Train-accuracy=0.023768
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [80]	Speed: 3.10 samples/sec	Train-accuracy=0.024691
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [90]	Speed: 3.27 samples/sec	Train-accuracy=0.028846
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [100]	Speed: 3.42 samples/sec	Train-accuracy=0.033416
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [110]	Speed: 3.46 samples/sec	Train-accuracy=0.034910
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [120]	Speed: 3.44 samples/sec	Train-accuracy=0.040806
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [130]	Speed: 3.41 samples/sec	Train-accuracy=0.043893
;; INFO  ml.dmlc.mxnet.Callback$Speedometer: Epoch[0] Batch [140]	Speed: 3.42 samples/sec	Train-accuracy=0.045213
)

