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

(ns org.apache.clojure-mxnet.io
  (:refer-clojure :exclude [next])
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [clojure.spec.alpha :as s]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random])
  (:import (org.apache.mxnet IO DataDesc DataBatch NDArray)
           (org.apache.mxnet.io ResizeIter PrefetchingIter NDArrayIter MXDataIter)))

(defn batches
  "Convert the data-pack to a batch seq"
  [data-pack]
  (util/scala-iterator->seq (.toIterator data-pack)))

(defn batch-label
  "Returns the vector of ndarrays that represents the label"
  [batch]
  (util/scala-vector->vec (.label batch)))

(defn batch-data
  "Returns the vector of ndarrays that represents the data"
  [batch]
  (util/scala-vector->vec (.data batch)))

(defn batch-index
  "Returns the vector of ints that represents the index"
  [batch]
  (util/scala-vector->vec (.index batch)))

(defn batch-pad
  "Returns the pad of the batch"
  [batch]
  (.pad batch))

(defn iterator [data-pack]
  (.iterator data-pack))

(defn resize-iter [iter nbatch])

(defn provide-data [pack-iterator]
  (->> pack-iterator
       (.provideData)
       (util/scala-map->map)
       (mapv (fn [[k v]] {:name k :shape (mx-shape/->vec v)}))))

(defn provide-label [pack-iterator]
  (->> pack-iterator
       (.provideLabel)
       (util/scala-map->map)
       (mapv (fn [[k v]] {:name k :shape (mx-shape/->vec v)}))))

(defn reset [iterator]
  (.reset iterator))

(defn has-next? [iterator]
  (.hasNext iterator))

(defn next [iterator]
  (.next iterator))

(defn iter-label [iterator]
  (util/scala-vector->vec (.getLabel iterator)))

(defn iter-data [iterator]
  (util/scala-vector->vec (.getData iterator)))

(defn iter-init-label [iterator]
  (util/scala-vector->vec (.initLabel iterator)))

(defmacro do-batches [iter f]
  "Takes an iterator and a function of one argument. The iterator will be reset and run thhrough all the batches with the batch passed to the function argument. nil is returned"
  `(do
     (reset ~iter)
     (loop [it# ~iter]
       (when (has-next? it#)
         (let [b# (next it#)]
           (do (~f b#))
           (recur it#))))))

(defmacro for-batches
  "Takes an iterator and a function of one argument. The iterator will be reset and run thhrough all the batches with the batch passed to the function argument. The result of the function will be conjed to a vector result of all the batches and returned at the end."
  [iter f]
  `(do
     (reset ~iter)
     (loop [it# ~iter
            result# []]
       (if (has-next? it#)
         (let [b# (next it#)]
           (recur it# (conj result# (do (~f b#)))))
         result#))))

(defmacro reduce-batches
  "Takes an iterator and a function of two arguments. The iterator will be reset and run thhrough all the batches with the batch passed to the function argument. The result of the function will the result of the reduce function"
  ([iter f initial-val]
   `(do
      (reset ~iter)
      (loop [it# ~iter
             result# ~initial-val]
        (if (has-next? it#)
          (let [b# (next it#)
                r# (do (~f result# b#))]
            (recur it# r#))
          result#))))
  ([iter f]
   `(reduce-batches ~iter ~f 0)))

(defn
  csv-iter
  ([kwargs]
   (util/apply-scala-fn (IO/CSVIter) (util/convert-io-map kwargs))))

(defn
  csv-pack
  ([kwargs]
   (util/apply-scala-fn (IO/CSVPack) (util/convert-io-map kwargs))))

(defn
  image-recode-pack
  ([kwargs]
   (util/apply-scala-fn
    (IO/ImageRecodePack)
    (util/convert-io-map kwargs))))

(defn
  image-record-iter
  ([kwargs]
   (util/apply-scala-fn
    (IO/ImageRecordIter)
    (util/convert-io-map kwargs))))

(defn
  mnist-iter
  ([kwargs]
   (util/apply-scala-fn (IO/MNISTIter) (util/convert-io-map kwargs))))

(defn
  mnist-pack
  ([kwargs]
   (util/apply-scala-fn (IO/MNISTPack) (util/convert-io-map kwargs))))

(defn
  create-iterator
  ([iter-name kwargs-map]
   (util/coerce-return (IO/createIterator iter-name (util/convert-io-map kwargs-map)))))

(defn
  create-mx-data-pack
  ([pack-name kwargs-map]
   (util/coerce-return (IO/createMXDataPack pack-name (util/convert-io-map kwargs-map)))))

(defn resize-iter
  "* Resize a data iterator to given number of batches per epoch.
  * May produce incomplete batch in the middle of an epoch due
  * to padding from internal iterator.
  *
  * @param data-iter Internal data iterator.
  * @param resize number of batches per epoch to resize to.
  * @param reset-internal whether to reset internal iterator with reset"
  [data-iter resize reset-iternal]
  (new ResizeIter data-iter resize reset-iternal))

(defn prefetching-iter
  "Takes one or more data iterators and combines them with pre-fetching"
  [iters data-names label-names]
  (new PrefetchingIter
       (util/vec->indexed-seq iters)
       (->> data-names
            (mapv util/convert-map)
            (util/vec->indexed-seq))
       (->> label-names
            (mapv util/convert-map)
            (util/vec->indexed-seq))))

(defn ndarray-iter
  " * NDArrayIter object in mxnet. Taking NDArray to get dataiter.
  *
  * @param data vector of iter
  * @opts map of:
  *     :label Same as data, but is not fed to the model during testing.
  *     :data-batch-size Batch Size (default 1)
  *     :shuffle Whether to shuffle the data (default false)
  *     :last-batch-handle = pad, discard, or rollover. (default pad)
  *     :data-name String of data name (default data)
  *     :label-name String of label name (default label)
  *  How to handle the last batch
  * This iterator will pad, discard or roll over the last batch if
  * the size of data does not match batch-size. Roll over is intended
  * for training and can cause problems if used for prediction."
  ([data {:keys [label data-batch-size shuffle last-batch-handle data-name label-name] :as opts
          :or {label nil
               data-batch-size 1
               shuffle false
               last-batch-handle "pad"
               data-name "data"
               label-name "label"}}]
   (new NDArrayIter
        (util/vec->indexed-seq data)
        (if label (util/vec->indexed-seq label) (util/empty-indexed-seq))
        (int data-batch-size)
        shuffle
        last-batch-handle
        data-name
        label-name))
  ([data]
   (ndarray-iter data {})))

(defn dispose [iterator]
  (.dispose iterator))

(s/def ::name string?)
(s/def ::shape vector?)
(s/def ::dtype #{dtype/UINT8 dtype/INT32 dtype/FLOAT16 dtype/FLOAT32 dtype/FLOAT64})
(s/def ::data-desc (s/keys :req-un [::name ::shape] :opt-un [::dtype ::layout]))

;; NCHW is N:batch size C: channel H: height W: width
;;; other layouts are
;; NT, TNC, nad N
;; the shape length must match the lengh of the layout string size
(defn data-desc
  ([{:keys [name shape dtype layout] :as opts
     :or {dtype base/MX_REAL_TYPE}}]
   (util/validate! ::data-desc opts "Invalid data description")
   (let [sc (count shape)
         layout (or layout (cond
                             (= 1 sc) "N"
                             (= 2 sc) "NT"
                             (= 3 sc) "TNC"
                             (= 4 sc) "NCHW"
                             :else (apply str (repeat sc "?"))))]
     (new DataDesc name (mx-shape/->shape shape) dtype layout)))
  ([name shape]
   (data-desc {:name name :shape shape})))

(s/def ::ndarray #(instance? NDArray %))
(s/def ::data vector?)
(s/def ::label (s/nilable (s/coll-of ::ndarray :kind vector?)))
(s/def ::index (s/nilable (s/coll-of int? :kind vector?)))
(s/def ::pad integer?)
(s/def ::bucket-key string?)
(s/def ::provided-data ::data-desc)
(s/def ::provided-label ::data-desc)
(s/def ::data-batch-class #(instance? DataBatch %))

(s/def ::data-batch
  (s/or
   :data-batch-class
   ::data-batch-class
   :data-batch-map
   (s/keys :req-un [::data] :opt-un [::label ::index ::pad ::bucket-key ::provided-data ::provided-label])))

(defn data-batch
  [{:keys [data label index pad bucket-key provided-data provided-label] :as info
    :or {data [] label [] index [] pad 0}}]
  ;;  provided-data and provided label is a map of name to shape to indicate the order of the data/label loading
  (util/validate! ::data-batch info "Invalid data batch")
  (new DataBatch
       (util/vec->indexed-seq data)
       (util/vec->indexed-seq label)
       (util/vec->indexed-seq index)
       (int pad)
       bucket-key
       (when provided-data (util/list-map provided-data))
       (when provided-label (util/list-map provided-label))))

(defn rand-iter
  "A implementation of a random noise iterator
   Instead of data pass in the shape vector of the noise shape"
  ([shape-vec {:keys [label data-batch-size shuffle last-batch-handle data-name label-name] :as opts
               :or {label nil
                    data-batch-size 1
                    shuffle false
                    last-batch-handle "pad"
                    data-name "rand"
                    label-name "label"}}]
   (let [data [(ndarray/ones shape-vec)]]
     (proxy [NDArrayIter]
            [(util/vec->indexed-seq data)
             (if label (util/vec->indexed-seq label) (util/empty-indexed-seq))
             (int data-batch-size)
             shuffle
             last-batch-handle
             data-name
             label-name]
       (provideData []
         (util/list-map {data-name (mx-shape/->vec (ndarray/shape (first data)))}))
       (provideLabel [] (util/empty-list-map))
       (hasNext [] true)
       (getData
         ([] (util/vec->indexed-seq [(random/normal 0 1 (mx-shape/->vec (ndarray/shape (first data))))])))
       (getLabel
         ([] (util/vec->indexed-seq []))))))
  ([shape-vec]
   (rand-iter shape-vec {})))
