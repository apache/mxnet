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

(ns neural-style.model-vgg-19
  (:require [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]))

(defn get-symbol []
  (let [data (sym/variable "data")
        conv1-1 (sym/convolution "conv1_1" {:data data :num-filter 64 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu1-1 (sym/activation "relu1_1" {:data conv1-1 :act-type "relu"})
        conv1-2 (sym/convolution "conv1-2" {:data relu1-1 :num-filter 64 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu1-2 (sym/activation "relu1_2" {:data conv1-2 :act-type "relu"})
        pool1 (sym/pooling "pool1" {:data relu1-2 :pad [0 0] :kernel [2 2] :stride [2 2] :pool-type "avg"})
        conv2-1 (sym/convolution "conv2_1" {:data pool1 :num-filter 128 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu2-1 (sym/activation "relu2_1" {:data conv2-1 :act-type "relu"})
        conv2-2 (sym/convolution "conv2_2" {:data relu2-1 :num-filter 128 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu2-2 (sym/activation "relu2_2" {:data conv2-2 :act-type "relu"})
        pool2 (sym/pooling "pool2" {:data relu2-2 :pad [0 0] :kernel [2 2] :stride [2 2] :pool-type "avg"})
        conv3-1 (sym/convolution "conv3_1" {:data pool2 :num-filter 256 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu3-1 (sym/activation "relu3_1" {:data conv3-1 :act-type "relu"})
        conv3-2 (sym/convolution "conv3_2" {:data relu3-1 :num-filter 256 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu3-2 (sym/activation "relu3_2" {:data conv3-2 :act-type "relu"})
        conv3-3 (sym/convolution "conv3_3" {:data relu3-2 :num-filter 256 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu3-3 (sym/activation "relu3_3" {:data conv3-3 :act-type "relu"})
        conv3-4 (sym/convolution "conv3_4" {:data relu3-3 :num-filter 256 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu3-4 (sym/activation "relu3_4" {:data conv3-4 :act-type "relu"})
        pool3 (sym/pooling "pool3" {:data relu3-4 :pad [0 0] :kernel [2 2] :stride [2 2] :pool-type "avg"})
        conv4-1 (sym/convolution "conv4_1" {:data pool3 :num-filter 512 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu4-1 (sym/activation "relu4_1" {:data conv4-1 :act-type "relu"})
        conv4-2 (sym/convolution "conv4_2" {:data relu4-1 :num-filter 512 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu4-2 (sym/activation "relu4_2" {:data conv4-2 :act-type "relu"})
        conv4-3 (sym/convolution "conv4_3" {:data relu4-2 :num-filter 512 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu4-3 (sym/activation "relu4_3" {:data conv4-3 :act-type "relu"})
        conv4-4 (sym/convolution "conv4_4" {:data relu4-3 :num-filter 512 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu4-4 (sym/activation "relu4_4" {:data conv4-4 :act-type "relu"})
        pool4 (sym/pooling "pool4" {:data relu4-4 :pad [0 0] :kernel [2 2] :stride [2 2] :pool-type "avg"})
        conv5-1 (sym/convolution "conv5_1" {:data pool4 :num-filter 512 :pad [1 1] :kernel [3 3] :stride [1 1]
                                            :no-bias false :workspace 1024})
        relu5-1 (sym/activation "relu5_1" {:data conv5-1 :act-type "relu"})

        ;;; style and content layers
        style (sym/group [relu1-1 relu2-1 relu3-1 relu4-1 relu5-1])
        content (sym/group [relu1-1])]
    {:style style :content content}))

(defn get-executor [style content model-path input-size ctx]
  (let [out (sym/group [style content])
        ;; make executor
        [arg-shapes output-shapes aux-shapes] (sym/infer-shape out {:data [1 3 (first input-size) (second input-size)]})
        arg-names (sym/list-arguments out)
        arg-map (zipmap arg-names (map #(ndarray/zeros % {:ctx ctx}) arg-shapes))
        grad-map {"data" (ndarray/copy-to (get arg-map "data") ctx)}
        ;; init with pre-training weights
        ;;; I'm not sure this is being set properly
        pretrained (do (ndarray/load model-path))
        arg-map (into {} (mapv (fn [[k v]]
                                 (let [pretrained-key (str "arg:" k)]
                                   (if (and (get pretrained pretrained-key) (not= "data" k))
                                     (do (ndarray/set v (get pretrained pretrained-key))
                                         [k v])
                                     [k v])))
                               arg-map))
        exec (sym/bind out ctx arg-map grad-map)
        outs (executor/outputs exec)]
    {:executor exec
     :data (get arg-map "data")
     :data-grad (get grad-map "data")
     :style (into [] (butlast outs))
     :content (last outs)
     :arg-map arg-map}))
