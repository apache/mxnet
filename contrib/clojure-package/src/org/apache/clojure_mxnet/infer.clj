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
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.shape :as shape]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (java.awt.image BufferedImage)
           (org.apache.mxnet NDArray)
           (org.apache.mxnet.infer Classifier ImageClassifier
                                   ObjectDetector Predictor)))

(s/def ::predictor #(instance? Predictor %))
(s/def ::classifier #(instance? Classifier %))
(s/def ::image-classifier #(instance? ImageClassifier %))
(s/def ::object-detector #(instance? ObjectDetector %))

(defrecord WrappedPredictor [predictor])
(defrecord WrappedClassifier [classifier])
(defrecord WrappedImageClassifier [image-classifier])
(defrecord WrappedObjectDetector [object-detector])

(s/def ::ndarray #(instance? NDArray %))
(s/def ::float-array (s/and #(.isArray (class %)) #(every? float? %)))
(s/def ::vec-of-float-arrays (s/coll-of ::float-array :kind vector?))
(s/def ::vec-of-ndarrays (s/coll-of ::ndarray :kind vector?))

(s/def ::wrapped-predictor (s/keys :req-un [::predictor]))
(s/def ::wrapped-classifier (s/keys :req-un [::classifier]))
(s/def ::wrapped-image-classifier (s/keys :req-un [::image-classifier]))
(s/def ::wrapped-detector (s/keys :req-un [::object-detector]))

(defprotocol APredictor
  (predict [wrapped-predictor inputs])
  (predict-with-ndarray [wrapped-predictor input-arrays]))

(defprotocol AClassifier
  (classify
    [wrapped-classifier inputs]
    [wrapped-classifier inputs topk])
  (classify-with-ndarray
    [wrapped-classifier inputs]
    [wrapped-classifier inputs topk]))

(defprotocol AImageClassifier
  (classify-image
    [wrapped-image-classifier image]
    [wrapped-image-classifier image topk])
  (classify-image-batch
    [wrapped-image-classifier images]
    [wrapped-image-classifier images topk]))

(defprotocol AObjectDetector
  (detect-objects
    [wrapped-detector image]
    [wrapped-detector image topk])
  (detect-objects-batch
    [wrapped-detector images]
    [wrapped-detector images topk])
  (detect-objects-with-ndarrays
    [wrapped-detector input-arrays]
    [wrapped-detector input-arrays topk]))

(extend-protocol APredictor
  WrappedPredictor
  (predict [wrapped-predictor inputs]
    (util/validate! ::wrapped-predictor wrapped-predictor
                    "Invalid predictor")
    (util/validate! ::vec-of-float-arrays inputs
                    "Invalid inputs")
    (util/coerce-return-recursive
     (.predict (:predictor wrapped-predictor)
               (util/vec->indexed-seq inputs))))
  (predict-with-ndarray [wrapped-predictor input-arrays]
    (util/validate! ::wrapped-predictor wrapped-predictor
                    "Invalid predictor")
    (util/validate! ::vec-of-ndarrays input-arrays
                    "Invalid input arrays")
    (util/coerce-return-recursive
     (.predictWithNDArray (:predictor wrapped-predictor)
                          (util/vec->indexed-seq input-arrays)))))

(s/def ::nil-or-int (s/nilable int?))

(extend-protocol AClassifier
  WrappedClassifier
  (classify [wrapped-classifier inputs]
    (util/validate! ::wrapped-classifier wrapped-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-float-arrays inputs
                    "Invalid inputs")
    (classify wrapped-classifier inputs nil))
  (classify [wrapped-classifier inputs topk]
    (util/validate! ::wrapped-classifier wrapped-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-float-arrays inputs
                    "Invalid inputs")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classify (:classifier wrapped-classifier)
                (util/vec->indexed-seq inputs)
                (util/->int-option topk))))
  (classify-with-ndarray [wrapped-classifier inputs]
    (util/validate! ::wrapped-classifier wrapped-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-ndarrays inputs
                    "Invalid inputs")
    (classify-with-ndarray wrapped-classifier inputs nil))
  (classify-with-ndarray [wrapped-classifier inputs topk]
    (util/validate! ::wrapped-classifier wrapped-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-ndarrays inputs
                    "Invalid inputs")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classifyWithNDArray (:classifier wrapped-classifier)
                           (util/vec->indexed-seq inputs)
                           (util/->int-option topk))))
  WrappedImageClassifier
  (classify [wrapped-image-classifier inputs]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-float-arrays inputs
                    "Invalid inputs")
    (classify wrapped-image-classifier inputs nil))
  (classify [wrapped-image-classifier inputs topk]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-float-arrays inputs
                    "Invalid inputs")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classify (:image-classifier wrapped-image-classifier)
                (util/vec->indexed-seq inputs)
                (util/->int-option topk))))
  (classify-with-ndarray [wrapped-image-classifier inputs]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-ndarrays inputs
                    "Invalid inputs")
    (classify-with-ndarray wrapped-image-classifier inputs nil))
  (classify-with-ndarray [wrapped-image-classifier inputs topk]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::vec-of-ndarrays inputs
                    "Invalid inputs")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classifyWithNDArray (:image-classifier wrapped-image-classifier)
                           (util/vec->indexed-seq inputs)
                           (util/->int-option topk)))))

(s/def ::image #(instance? BufferedImage %))

(extend-protocol AImageClassifier
  WrappedImageClassifier
  (classify-image [wrapped-image-classifier image]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::image image "Invalid image")
    (classify-image wrapped-image-classifier image nil))
  (classify-image [wrapped-image-classifier image topk]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::image image "Invalid image")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classifyImage (:image-classifier wrapped-image-classifier)
                     image
                     (util/->int-option topk))))
  (classify-image-batch [wrapped-image-classifier images]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (classify-image-batch wrapped-image-classifier images nil))
  (classify-image-batch [wrapped-image-classifier images topk]
    (util/validate! ::wrapped-image-classifier wrapped-image-classifier
                    "Invalid classifier")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.classifyImageBatch (:image-classifier wrapped-image-classifier)
                          images
                          (util/->int-option topk)))))

(extend-protocol AObjectDetector
  WrappedObjectDetector
  (detect-objects [wrapped-detector image]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (util/validate! ::image image "Invalid image")
    (detect-objects wrapped-detector image nil))
  (detect-objects [wrapped-detector image topk]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (util/validate! ::image image "Invalid image")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.imageObjectDetect (:object-detector wrapped-detector)
                         image
                         (util/->int-option topk))))
  (detect-objects-batch [wrapped-detector images]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (detect-objects-batch wrapped-detector images nil))
  (detect-objects-batch [wrapped-detector images topk]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.imageBatchObjectDetect (:object-detector wrapped-detector)
                              images
                              (util/->int-option topk))))
  (detect-objects-with-ndarrays [wrapped-detector input-arrays]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (util/validate! ::vec-of-ndarrays input-arrays
                    "Invalid inputs")
    (detect-objects-with-ndarrays wrapped-detector input-arrays nil))
  (detect-objects-with-ndarrays [wrapped-detector input-arrays topk]
    (util/validate! ::wrapped-detector wrapped-detector
                    "Invalid object detector")
    (util/validate! ::vec-of-ndarrays input-arrays
                    "Invalid inputs")
    (util/validate! ::nil-or-int topk "Invalid top-K")
    (util/coerce-return-recursive
     (.objectDetectWithNDArray (:object-detector wrapped-detector)
                               (util/vec->indexed-seq input-arrays)
                               (util/->int-option topk)))))

(defprotocol AInferenceFactory
  (create-predictor [factory] [factory opts])
  (create-classifier [factory] [factory opts])
  (create-image-classifier [factory] [factory opts])
  (create-object-detector [factory] [factory opts]))

(defn convert-descriptors
  [descriptors]
  (util/vec->indexed-seq
   (into [] (map mx-io/data-desc descriptors))))

(defrecord InferenceFactory [model-path-prefix input-descriptors]
  AInferenceFactory
  (create-predictor
    [factory]
    (create-predictor factory {}))
  (create-predictor
    [factory opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedPredictor
       (new Predictor
            model-path-prefix
            (convert-descriptors input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-classifier
    [factory]
    (create-classifier factory {}))
  (create-classifier
    [factory opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedClassifier
       (new Classifier
            model-path-prefix
            (convert-descriptors input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-image-classifier
    [factory]
    (create-image-classifier factory {}))
  (create-image-classifier
    [factory opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedImageClassifier
       (new ImageClassifier
            model-path-prefix
            (convert-descriptors input-descriptors)
            (into-array contexts)
            (util/->int-option epoch)))))
  (create-object-detector
    [factory]
    (create-object-detector factory {}))
  (create-object-detector
    [factory opts]
    (let [{:keys [contexts epoch]
           :or {contexts [(context/cpu)] epoch 0}} opts]
      (->WrappedObjectDetector
       (new ObjectDetector
            model-path-prefix
            (convert-descriptors input-descriptors)
            (into-array contexts)
            (util/->int-option epoch))))))

(s/def ::model-path-prefix string?)
(s/def ::input-descriptors (s/coll-of ::mx-io/data-desc))

(defn model-factory
  "Creates a factory that can be used to instantiate an image classifier
  predictor or object detector"
  [model-path-prefix input-descriptors]
  (util/validate! ::model-path-prefix model-path-prefix
                  "Invalid model path prefix")
  (util/validate! ::input-descriptors input-descriptors
                  "Invalid input descriptors")
  (->InferenceFactory model-path-prefix input-descriptors))

(defn reshape-image
  "Reshape an image to a new shape"
  [image width height]
  (util/validate! ::image image "Invalid image")
  (util/validate! int? width "Invalid width")
  (util/validate! int? height "Invalid height")
  (ImageClassifier/reshapeImage image width height))

(defn buffered-image-to-pixels
  "Convert input BufferedImage to NDArray of input shape"
  [image input-shape-vec]
  (util/validate! ::image image "Invalid image")
  (util/validate! (s/coll-of int?) input-shape-vec "Invalid shape vector")
  (ImageClassifier/bufferedImageToPixels image (shape/->shape input-shape-vec)))

(s/def ::image-path string?)
(s/def ::image-paths (s/coll-of ::image-path))

(defn load-image-from-file
  "Loads an input image given a file name"
  [image-path]
  (util/validate! ::image-path image-path "Invalid image path")
  (ImageClassifier/loadImageFromFile image-path))

(defn load-image-paths
  "Loads images from a list of file names"
  [image-paths]
  (util/validate! ::image-paths image-paths "Invalid image paths")
  (ImageClassifier/loadInputBatch (util/convert-vector image-paths)))
