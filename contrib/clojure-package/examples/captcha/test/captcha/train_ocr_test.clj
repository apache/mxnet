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

(ns captcha.train-ocr-test
  (:require [clojure.test :refer :all]
            [captcha.consts :refer :all]
            [captcha.train-ocr :refer :all]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as shape]
            [org.apache.clojure-mxnet.util :as util]))

(deftest test-consts
  (is (= 8 batch-size))
  (is (= [3 30 80] data-shape))
  (is (= 4 label-width))
  (is (= 10 num-labels)))

(deftest test-labeled-data
  (let [train-batch (mx-io/next train-data)
        eval-batch (mx-io/next eval-data)
        allowed-labels (into #{} (map float (range 10)))]
    (is (= 8 (-> train-batch mx-io/batch-index count)))
    (is (= 8 (-> eval-batch mx-io/batch-index count)))
    (is (= [8 3 30 80] (-> train-batch
                           mx-io/batch-data
                           first
                           ndarray/shape-vec)))
    (is (= [8 3 30 80] (-> eval-batch
                           mx-io/batch-data
                           first
                           ndarray/shape-vec)))
    (is (every? #(<= 0 % 255) (-> train-batch
                                  mx-io/batch-data
                                  first
                                  ndarray/->vec)))
    (is (every? #(<= 0 % 255) (-> eval-batch
                                  mx-io/batch-data
                                  first
                                  ndarray/->vec)))
    (is (= [8 4] (-> train-batch
                     mx-io/batch-label
                     first
                     ndarray/shape-vec)))
    (is (= [8 4] (-> eval-batch
                     mx-io/batch-label
                     first
                     ndarray/shape-vec)))
    (is (every? allowed-labels (-> train-batch
                                   mx-io/batch-label
                                   first
                                   ndarray/->vec)))
    (is (every? allowed-labels (-> eval-batch
                                   mx-io/batch-label
                                   first
                                   ndarray/->vec)))))

(deftest test-model
  (let [batch (mx-io/next train-data)
        model (m/module (create-captcha-net)
                        {:data-names ["data"] :label-names ["label"]})
        _ (m/bind model
                  {:data-shapes (mx-io/provide-data-desc train-data)
                   :label-shapes (mx-io/provide-label-desc train-data)})
        _ (m/init-params model)
        _ (m/forward-backward model batch)
        output-shapes (-> model
                          m/output-shapes
                          util/coerce-return-recursive)
        outputs (-> model
                    m/outputs-merged
                    first)
        grads (->> model m/grad-arrays (map first))]
    (is (= [["softmaxoutput0_output" (shape/->shape [8 10])]]
           output-shapes))
    (is (= [32 10] (-> outputs ndarray/shape-vec)))
    (is (every? #(<= 0.0 % 1.0) (-> outputs ndarray/->vec)))
    (is (= [[32 3 5 5] [32]   ; convolution1 weights+bias
            [32 32 5 5] [32]  ; convolution2 weights+bias
            [32 32 3 3] [32]  ; convolution3 weights+bias
            [32 32 3 3] [32]  ; convolution4 weights+bias
            [256 28672] [256] ; fully-connected1 weights+bias
            [10 256] [10]     ; 1st label scores
            [10 256] [10]     ; 2nd label scores
            [10 256] [10]     ; 3rd label scores
            [10 256] [10]]    ; 4th label scores
           (map ndarray/shape-vec grads)))))

(deftest test-accuracy
  (let [labels (ndarray/array [1 2 3 4,
                               5 6 7 8]
                              [2 4])
        pred-labels (ndarray/array [1 0,
                                    2 6,
                                    3 0,
                                    4 8]
                                   [8])
        preds (ndarray/one-hot pred-labels 10)]
    (is (float? (accuracy labels preds)))
    (is (float? (accuracy labels preds :by-character false)))
    (is (float? (accuracy labels preds :by-character true)))
    (is (= 0.5 (accuracy labels preds)))
    (is (= 0.5 (accuracy labels preds :by-character false)))
    (is (= 0.75 (accuracy labels preds :by-character true)))))
