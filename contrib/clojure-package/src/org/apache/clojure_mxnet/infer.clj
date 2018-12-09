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
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.io :as io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet NDArray)
           (org.apache.mxnet.infer Classifier ImageClassifier
                                   ObjectDetector Predictor)))

(defprotocol APredictor
  (predict [this input])
  (predict-with-ndarray [this input-array]))

(defprotocol AClassifier
  (classify [this input] [this input topk])
  (classify-with-ndarray [this input] [this input topk]))

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
  (predict [this input]
    (util/coerce-return-recursive
     (.predict (:predictor this) input)))
  (predict-with-ndarray [this input-array]
    (util/coerce-return-recursive
     (.predictWithNDArray (:predictor this) input-array))))

(extend-protocol AClassifier
  WrappedClassifier
  (classify [this input]
    (classify this input nil))
  (classify [this input topk]
    (util/coerce-return-recursive
     (.classify (:classifier this) input (util/->int-option topk))))
  (classify-with-ndarray [this input]
    (classify-with-ndarray this input nil))
  (classify-with-ndarray [this input topk]
    (util/coerce-return-recursive
     (.classifyWithNDArray (:classifier this) input (util/->int-option topk))))
  WrappedImageClassifier
  (classify [this input]
    (classify this input nil))
  (classify [this input topk]
    (util/coerce-return-recursive
     (.classify (:image-classifier this) input (util/->int-option topk))))
  (classify-with-ndarray [this input]
    (classify-with-ndarray this input nil))
  (classify-with-ndarray [this input topk]
    (util/coerce-return-recursive
     (.classifyWithNDArray (:image-classifier this)
                           input
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
                              (util/convert-vector images)
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

;;; Testing
;(def factory (->InferenceFactory
;              "/Users/kedar_bellare/Projects/incubator-mxnet/contrib/clojure-package/scripts/infer/models/resnet-152/resnet-152"
;              [(io/data-desc {:name "data"
;                              :shape [1 3 224 224]
;                              :layout layout/NCHW
;                              :dtype dtype/FLOAT32})]))
;(def imcls (create-image-classifier factory))
(def factory (->InferenceFactory
              "/Users/kedar_bellare/Projects/incubator-mxnet/contrib/clojure-package/scripts/infer/models/resnet50_ssd/resnet50_ssd_model"
              [(io/data-desc {:name "data"
                             :shape [1 3 512 512]
                              :layout layout/NCHW
                              :dtype dtype/FLOAT32})]))
(def imdetect (create-object-detector factory))

(def kitten-file "/Users/kedar_bellare/Projects/incubator-mxnet/contrib/clojure-package/scripts/infer/images/kitten.jpg")
(def kitten (load-image-from-file kitten-file))
(def kittens (load-image-paths [kitten-file kitten-file]))

;(classify-image imcls kitten 2)
;(classify-image-batch imcls kittens 2)
(detect-objects imdetect kitten 2)
