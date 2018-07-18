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

(ns org.apache.clojure-mxnet.io-test
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [clojure.test :refer :all]))

(deftest test-mnsit-iter-and-mnist-pack
  (let [_ (when-not (.exists (io/file "data/train-images-idx3-ubyte"))
            (sh "scripts/get_mnist_data.sh"))
        params {:image "data/train-images-idx3-ubyte"
                :label "data/train-labels-idx1-ubyte"
                :data-shape [784]
                :batch-size 100
                :shuffle 1
                :flat 1
                :silent 0
                :seed 10}
        mnist-pack (mx-io/mnist-pack params)]
    (is (= 600 (count (mx-io/batches mnist-pack))))

    (let [mnist-iter (mx-io/iterator mnist-pack)
          provide-data (mx-io/provide-data mnist-iter)
          provide-label (mx-io/provide-label mnist-iter)]
      (is (= [100 784] (-> provide-data first :shape)))
      (is (= [100] (->  provide-label first :shape)))
      (is (= 600 (mx-io/reduce-batches mnist-iter (fn [result batch] (inc result)))))
      ;; test reset
      (let [_ (mx-io/reset mnist-iter)
            _ (mx-io/next mnist-iter)
            label0 (-> (mx-io/iter-label mnist-iter) first (ndarray/->vec))
            data0 (-> (mx-io/iter-data mnist-iter) first (ndarray/->vec))
            _ (mx-io/next mnist-iter)
            _ (mx-io/next mnist-iter)
            _ (mx-io/next mnist-iter)
            _ (mx-io/reset mnist-iter)
            _ (mx-io/next mnist-iter)
            label1 (-> (mx-io/iter-label mnist-iter) first (ndarray/->vec))
            data1 (-> (mx-io/iter-data mnist-iter) first (ndarray/->vec))]
        (is (= label1 label0))
        (is (= data1 data0))))))

(deftest test-image-record-iter
  (let [_ (when-not (.exists (io/file "data/cifar/train.rec"))
            (sh "scripts/get_cifar_data.sh"))
        params {:path-imgrec "data/cifar/train.rec"
                :label "data/cifar/cifar10_mean.bin"
                :rand-crop false
                :and-mirror false
                :shuffle false
                :data-shape [3 28 28]
                :batch-size 100
                :preprocess-threads 4
                :prefetch-buffer 1}
        img-rec-iter (mx-io/image-record-iter params)
        nbatch 500]
    (is (= [100 3 28 28] (-> (mx-io/provide-data img-rec-iter) first :shape)))
    (is (= [100] (->  (mx-io/provide-label img-rec-iter) first :shape)))
    (is (= nbatch (mx-io/reduce-batches img-rec-iter (fn [result batch] (inc result)))))))

(deftest test-resize-iter
  (let [_ (when-not (.exists (io/file "data/train-images-idx3-ubyte"))
            (sh "scripts/get_mnist_data.sh"))
        params {:image "data/train-images-idx3-ubyte"
                :label "data/train-labels-idx1-ubyte"
                :data-shape [784]
                :batch-size 100
                :shuffle 1
                :flat 1
                :silent 0
                :seed 10}
        mnist-iter (mx-io/mnist-iter params)
        nbatch 400
        resize-iter (mx-io/resize-iter mnist-iter nbatch false)]
    (is (= nbatch (mx-io/reduce-batches resize-iter (fn [result batch] (inc result)))))
    (mx-io/reset resize-iter)
    (is (= nbatch (mx-io/reduce-batches resize-iter (fn [result batch] (inc result)))))))

(deftest test-prefetching-iter
  (let [_ (when-not (.exists (io/file "data/train-images-idx3-ubyte"))
            (sh "scripts/get_mnist_data.sh"))
        params {:image "data/train-images-idx3-ubyte"
                :label "data/train-labels-idx1-ubyte"
                :data-shape [784]
                :batch-size 100
                :shuffle 1
                :flat 1
                :silent 0
                :seed 10}
        mnist-iter1 (mx-io/mnist-iter params)
        mnist-iter2 (mx-io/mnist-iter params)
        nbatch 600
        prefetch-iter (mx-io/prefetching-iter [mnist-iter1 mnist-iter2]
                                              [{"data" "data1"} {"data" "data2"}]
                                              [{"label" "label1"} {"label" "label2"}])]
    (is (= nbatch (mx-io/reduce-batches prefetch-iter (fn [result batch] (inc result)))))
    (let [provide-data (mx-io/provide-data prefetch-iter)
          provide-label (mx-io/provide-label prefetch-iter)]
      (is (= #{[100 784]} (into #{} (map :shape provide-data))))
      (is (= #{[100]} (into #{} (map :shape provide-label))))
      (mx-io/dispose prefetch-iter))))

(deftest test-ndarray-iter
  (let [shape0 [1000 2 2]
        data [(ndarray/ones shape0) (ndarray/zeros shape0)]
        shape1 [1000 1]
        label [(ndarray/ones shape1)]
        batch-data0 (ndarray/ones [128 2 2])
        batch-data1 (ndarray/zeros [128 2 2])
        batch-label (ndarray/ones [128 1])]

    ;; test pad
    (let [data-iter0 (mx-io/ndarray-iter data {:label label
                                               :data-batch-size 128
                                               :shuffle false
                                               :last-batch-handle "pad"})
          nbatch0 8]
      (is (= nbatch0 (count (mx-io/for-batches data-iter0 (fn [batch] 1)))))
      (is (every? true? (mx-io/for-batches data-iter0
                                           (fn [batch]
                                             (= batch-data0
                                                (first (mx-io/batch-data batch)))))))
      (is (every? true? (mx-io/for-batches data-iter0
                                           (fn [batch]
                                             (= batch-data1
                                                (second (mx-io/batch-data batch)))))))
      (is (every? true? (mx-io/for-batches data-iter0
                                           (fn [batch]
                                             (= batch-label
                                                (first (mx-io/batch-label batch))))))))

    ;; test discard
    (let [data-iter1 (mx-io/ndarray-iter data {:label label
                                               :data-batch-size 128
                                               :shuffle false
                                               :last-batch-handle "discard"})
          nbatch1 7]
      (is (= nbatch1 (mx-io/reduce-batches data-iter1 (fn [result batch] (inc result))))))

    ;; test empty label for prediction
    (let [data-iter2 (mx-io/ndarray-iter data {:data-batch-size 128
                                               :shuffle false
                                               :last-batch-handle "discard"})
          nbatch2 7]
      (is (= nbatch2 (mx-io/reduce-batches data-iter2 (fn [result batch] (inc result)))))
      (is (= [] (mx-io/iter-init-label data-iter2))))))
