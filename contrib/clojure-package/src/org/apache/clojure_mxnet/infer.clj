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

(ns org.apache.clojure-mxnet.infer
  (:refer-clojure :exclude [type])
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet NDArray)
           (org.apache.mxnet.infer Classifier ImageClassifier
                                   ObjectDetector Predictor)))

(defprotocol APredictor
  (predict [this inputs])
  (predict-with-ndarray [this input-arrays]))

(defprotocol AClassifier
  (classify [this inputs] [this inputs topk])
  (classify-with-ndarray [this inputs] [this inputs topk]))

(defprotocol AImageClassifier
  (classify-image [this image] [this image topk])
  (classify-image-batch [this images] [this images topk]))

(defprotocol AObjectDetector
  (detect-objects [this image] [this image topk])
  (detect-objects-batch [this images] [this images topk])
  (detect-objects-with-ndarrays
    [this input-arrays]
    [this input-arrays topk]))

(defrecord WrappedPredictor [predictor])
(defrecord WrappedClassifier [classifier])
(defrecord WrappedImageClassifier [image-classifier])
(defrecord WrappedObjectDetector [object-detector])

(extend-protocol APredictor
  WrappedPredictor
  (predict [this inputs]
    (util/coerce-return-recursive
     (.predict (:predictor this)
               (util/vec->indexed-seq inputs))))
  (predict-with-ndarray [this input-arrays]
    (util/coerce-return-recursive
     (.predictWithNDArray (:predictor this)
                          (util/vec->indexed-seq input-arrays)))))

(extend-protocol AClassifier
  WrappedClassifier
  (classify [this inputs]
    (classify this inputs nil))
  (classify [this inputs topk]
    (util/coerce-return-recursive
     (.classify (:classifier this)
                (util/vec->indexed-seq inputs)
                (util/->int-option topk))))
  (classify-with-ndarray [this inputs]
    (classify-with-ndarray this inputs nil))
  (classify-with-ndarray [this inputs topk]
    (util/coerce-return-recursive
     (.classifyWithNDArray (:classifier this)
                           (util/vec->indexed-seq inputs)
                           (util/->int-option topk))))
  WrappedImageClassifier
  (classify [this inputs]
    (classify this inputs nil))
  (classify [this inputs topk]
    (util/coerce-return-recursive
     (.classify (:image-classifier this)
                (util/vec->indexed-seq inputs)
                (util/->int-option topk))))
  (classify-with-ndarray [this inputs]
    (classify-with-ndarray this inputs nil))
  (classify-with-ndarray [this inputs topk]
    (util/coerce-return-recursive
     (.classifyWithNDArray (:image-classifier this)
                           (util/vec->indexed-seq inputs)
                           (util/->int-option topk)))))

(extend-protocol AImageClassifier
  WrappedImageClassifier
  (classify-image [this image]
    (classify-image this image nil))
  (classify-image [this image topk]
    (util/coerce-return-recursive
     (.classifyImage (:image-classifier this) image (util/->int-option topk))))
  (classify-image-batch [this images]
    (classify-image-batch this images nil))
  (classify-image-batch [this images topk]
    (util/coerce-return-recursive
     (.classifyImageBatch (:image-classifier this)
                          images
                          (util/->int-option topk)))))

(extend-protocol AObjectDetector
  WrappedObjectDetector
  (detect-objects [this image]
    (detect-objects this image nil))
  (detect-objects [this image topk]
    (util/coerce-return-recursive
     (.imageObjectDetect (:object-detector this)
                         image
                         (util/->int-option topk))))
  (detect-objects-batch [this images]
    (detect-objects-batch this images nil))
  (detect-objects-batch [this images topk]
    (util/coerce-return-recursive
     (.imageBatchObjectDetect (:object-detector this)
                              images
                              (util/->int-option topk))))
  (detect-objects-with-ndarrays [this input-arrays]
    (detect-objects-with-ndarrays this input-arrays nil))
  (detect-objects-with-ndarrays [this input-arrays topk]
    (util/coerce-return-recursive
     (.objectDetectWithNDArray (:object-detector this)
                               (util/vec->indexed-seq input-arrays)
                               (util/->int-option topk)))))

(defprotocol AInferenceFactory
  (create-predictor [this] [this opts])
  (create-classifier [this] [this opts])
  (create-image-classifier [this] [this opts])
  (create-object-detector [this] [this opts]))

(defrecord InferenceFactory [model-path-prefix input-descriptors]
  AInferenceFactory
  (create-predictor [this]
    (create-predictor this {}))
  (create-predictor [this opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedPredictor
       (new Predictor
            model-path-prefix
            (util/vec->indexed-seq input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-classifier [this]
    (create-classifier this {}))
  (create-classifier [this opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedClassifier
       (new Classifier
            model-path-prefix
            (util/vec->indexed-seq input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-image-classifier [this]
    (create-image-classifier this {}))
  (create-image-classifier [this opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedImageClassifier
       (new ImageClassifier
            model-path-prefix
            (util/vec->indexed-seq input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-object-detector [this]
    (create-object-detector this {}))
  (create-object-detector [this opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedObjectDetector
       (new ObjectDetector
            model-path-prefix
            (util/vec->indexed-seq input-descriptors)
            (into-array contexts)
            (util/->int-option epoch))))))

(defn model-factory
  "Creates a factory that can be used to instantiate an image classifier
  predictor or object detector"
  [model-path-prefix input-descriptors]
  (->InferenceFactory model-path-prefix input-descriptors))

(defn reshape-image
  "Reshape an image to a new shape"
  [image width height]
  (ImageClassifier/reshapeImage image width height))

(defn buffered-image-to-pixels
  "Convert input BufferedImage to NDArray of input shape"
  [image input-shape]
  (ImageClassifier/bufferedImageToPixels image input-shape))

(defn load-image-from-file
  "Loads an input image given a file name"
  [image-path]
  (ImageClassifier/loadImageFromFile image-path))

(defn load-image-paths
  "Loads images from a list of file names"
  [image-paths]
  (ImageClassifier/loadInputBatch (util/convert-vector image-paths)))
