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

(ns captcha.train-ocr
  (:require [captcha.consts :refer :all]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym])
  (:gen-class))

(when-not (.exists (io/file "captcha_example/captcha_train.lst"))
  (sh "./get_data.sh"))

(defonce train-data
  (mx-io/image-record-iter {:path-imgrec "captcha_example/captcha_train.rec"
                            :path-imglist "captcha_example/captcha_train.lst"
                            :batch-size batch-size
                            :label-width label-width
                            :data-shape data-shape
                            :shuffle true
                            :seed 42}))

(defonce eval-data
  (mx-io/image-record-iter {:path-imgrec "captcha_example/captcha_test.rec"
                            :path-imglist "captcha_example/captcha_test.lst"
                            :batch-size batch-size
                            :label-width label-width
                            :data-shape data-shape}))

(defn accuracy
  [label pred & {:keys [by-character]
                 :or {by-character false} :as opts}]
  (let [[nr nc] (ndarray/shape-vec label)
        pred-context (ndarray/context pred)
        label-t (-> label
                    ndarray/transpose
                    (ndarray/reshape [-1])
                    (ndarray/as-in-context pred-context))
        pred-label (ndarray/argmax pred 1)
        matches (ndarray/equal label-t pred-label)
        [digit-matches] (-> matches
                            ndarray/sum
                            ndarray/->vec)
        [complete-matches] (-> matches
                               (ndarray/reshape [nc nr])
                               (ndarray/sum 0)
                               (ndarray/equal label-width)
                               ndarray/sum
                               ndarray/->vec)]
    (if by-character
      (float (/ digit-matches nr nc))
      (float (/ complete-matches nr)))))

(defn get-data-symbol
  []
  (let [data (sym/variable "data")
        ;; normalize the input pixels
        scaled (sym/div (sym/- data 127) 128)

        conv1 (sym/convolution {:data scaled :kernel [5 5] :num-filter 32})
        pool1 (sym/pooling {:data conv1 :pool-type "max" :kernel [2 2] :stride [1 1]})
        relu1 (sym/activation {:data pool1 :act-type "relu"})

        conv2 (sym/convolution {:data relu1 :kernel [5 5] :num-filter 32})
        pool2 (sym/pooling {:data conv2 :pool-type "avg" :kernel [2 2] :stride [1 1]})
        relu2 (sym/activation {:data pool2 :act-type "relu"})

        conv3 (sym/convolution {:data relu2 :kernel [3 3] :num-filter 32})
        pool3 (sym/pooling {:data conv3 :pool-type "avg" :kernel [2 2] :stride [1 1]})
        relu3 (sym/activation {:data pool3 :act-type "relu"})

        conv4 (sym/convolution {:data relu3 :kernel [3 3] :num-filter 32})
        pool4 (sym/pooling {:data conv4 :pool-type "avg" :kernel [2 2] :stride [1 1]})
        relu4 (sym/activation {:data pool4 :act-type "relu"})

        flattened (sym/flatten {:data relu4})
        fc1 (sym/fully-connected {:data flattened :num-hidden 256})
        fc21 (sym/fully-connected {:data fc1 :num-hidden num-labels})
        fc22 (sym/fully-connected {:data fc1 :num-hidden num-labels})
        fc23 (sym/fully-connected {:data fc1 :num-hidden num-labels})
        fc24 (sym/fully-connected {:data fc1 :num-hidden num-labels})]
    (sym/concat "concat" nil [fc21 fc22 fc23 fc24] {:dim 0})))

(defn get-label-symbol
  []
  (as-> (sym/variable "label") label
    (sym/transpose {:data label})
    (sym/reshape {:data label :shape [-1]})))

(defn create-captcha-net
  []
  (let [scores (get-data-symbol)
        labels (get-label-symbol)]
    (sym/softmax-output {:data scores :label labels})))

(def optimizer
  (optimizer/adam
   {:learning-rate 0.0002
    :wd 0.00001
    :clip-gradient 10}))

(defn train-ocr
  [devs]
  (println "Starting the captcha training ...")
  (let [model (m/module
               (create-captcha-net)
               {:data-names ["data"] :label-names ["label"]
                :contexts devs})]
    (m/fit model {:train-data train-data
                  :eval-data eval-data
                  :num-epoch 10
                  :fit-params (m/fit-params
                               {:kvstore "local"
                                :batch-end-callback
                                (callback/speedometer batch-size 100)
                                :initializer
                                (initializer/xavier {:factor-type "in"
                                                     :magnitude 2.34})
                                :optimizer optimizer
                                :eval-metric (eval-metric/custom-metric
                                              #(accuracy %1 %2)
                                              "accuracy")})})
    (println "Finished the fit")
    model))

(defn -main
  [& args]
  (let [[dev dev-num] args
        num-devices (Integer/parseInt (or dev-num "1"))
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range num-devices))
               (mapv #(context/cpu %) (range num-devices)))
        model (train-ocr devs)]
    (m/save-checkpoint model {:prefix model-prefix :epoch 0})))
