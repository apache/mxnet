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

(ns org.apache.clojure-mxnet.ndarray-test
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as ctx]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.ndarray :as ndarray :refer [->vec zeros ones += -= *= full shape]]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.test-util :as test-util]
            [clojure.test :refer :all]))

(deftest test->vec
  (is (= [0.0 0.0 0.0 0.0] (->vec (zeros [2 2])))))

(deftest test-to-array
  (is (= [0.0 0.0 0.0 0.0] (vec (ndarray/to-array (zeros [2 2]))))))

(deftest test-to-scalar
  (is (= 0.0 (ndarray/to-scalar (zeros [1]))))
  (is (= 1.0 (ndarray/to-scalar (ones [1]))))
  (is (thrown-with-msg? Exception #"The current array is not a scalar"
                        (ndarray/to-scalar (zeros [1 1])))))

(deftest test-size-and-shape
  (let [m (zeros [4 1])]
    (is (= (mx-shape/->shape [4 1]) (ndarray/shape m)))
    (is (= 4 (ndarray/size m)))))

(deftest test-dtype
  (is (= base/MX_REAL_TYPE (ndarray/dtype (zeros [3 2])))))

(deftest test-set-scalar-value
  (is (= [10.0 10.0] (-> (ndarray/empty [2 1])
                         (ndarray/set 10)
                         (->vec)))))

(deftest test-copy-from-vector
  (is (= [1.0 2.0 3.0 4.0] (-> (ndarray/empty [4 1])
                               (ndarray/set [1 2 3 4])
                               (->vec)))))

(deftest test-plus
  (let [ndzeros (zeros [2 1])
        ndones (ndarray/+ ndzeros 1)]
    (is (= [1.0 1.0] (->vec ndones)))
    (is (= [2.0 2.0] (->vec (ndarray/+ ndones 1))))
    (is (= [1.0 1.0] (->vec ndones)))
    ;;; += mutuates
    (is (= [2.0 2.0] (->vec (+= ndones 1))))
    (is (= [2.0 2.0] (->vec ndones)))))

(deftest test-minus
  (let [ndones (ones [2 1])
        ndzeros (ndarray/- ndones 1)]
    (is (= [0.0 0.0] (->vec ndzeros)))
    (is (= [-1.0 -1.0] (->vec (ndarray/- ndzeros 1))))
    (is (= [0.0 0.0] (->vec ndzeros)))
    ;;; += mutuates
    (is (= [-1.0 -1.0] (->vec (-= ndzeros 1))))
    (is (= [-1.0 -1.0] (->vec ndzeros)))))

(deftest test-multiplication
  (let [ndones (ones [2 1])
        ndtwos (ndarray/* ndones 2)]
    (is (= [2.0 2.0] (->vec ndtwos)))
    (is (= [1.0 1.0] (->vec (ndarray/* ndones ndones))))
    (is (= [4.0 4.0] (->vec (ndarray/* ndtwos ndtwos))))
    ;; *= mutates
    (is (= [4.0 4.0] (->vec (*= ndtwos ndtwos))))
    (is (= [4.0 4.0] (->vec ndtwos)))))

(deftest test-division
  (let [ndones (ones [2 1])
        ndzeros (ndarray/- ndones 1)
        ndhalves (ndarray/div ndones 2)]
    (is (= [0.5 0.5] (->vec ndhalves)))
    (is (= [1.0 1.0] (->vec (ndarray/div ndhalves ndhalves))))
    (is (= [1.0 1.0] (->vec (ndarray/div ndones ndones))))
    (is (= [0.0 0.0] (->vec (ndarray/div ndzeros ndones))))
    ;; div= mutates
    (is (= [1.0 1.0] (->vec (ndarray/div= ndhalves ndhalves))))
    (is (= [1.0 1.0] (->vec ndhalves)))))

(deftest test-full
  (let [nda (full [1 2] 3.0)]
    (is (= (shape nda) (mx-shape/->shape [1 2])))
    (is (= [3.0 3.0] (->vec nda)))))

(deftest test-clip
  (let [nda (-> (ndarray/empty [3 2])
                (ndarray/set [1 2 3 4 5 6]))]
    (is (= [2.0 2.0 3.0 4.0 5.0 5.0] (->vec (ndarray/clip nda 2 5))))))

(deftest test-sqrt
  (let [nda (-> (ndarray/empty [4 1])
                (ndarray/set [0 1 4 9]))]
    (is (= [0.0 1.0 2.0 3.0] (->vec (ndarray/sqrt nda))))))

(deftest test-rsqrt
  (let [nda (ndarray/array [1.0 4.0] [2 1])]
    (is (= [1.0 0.5] (->vec (ndarray/rsqrt nda))))))

(deftest test-norm
  (let [nda (-> (ndarray/empty [3 1])
                (ndarray/set [1 2 3]))
        normed (ndarray/norm nda)]
    (is (= [1] (mx-shape/->vec (shape normed))))
    (is (test-util/approx= 1e-4 (Math/sqrt 14.0) (ndarray/to-scalar normed)))))

(deftest test-one-hot-encode
  (let [nda1 (ndarray/array [1 0 2] [3])
        nda2 (ndarray/empty [3 3])
        res (ndarray/onehot-encode nda1 nda2)]
    (is (= [3 3] (mx-shape/->vec (shape res))))
    (is (= [0.0 1.0 0.0
            1.0 0.0 0.0
            0.0 0.0 1.0] (->vec res)))))

(deftest test-dot
  (let [nda1 (ndarray/array [1 2] [1 2])
        nda2 (ndarray/array [3 4] [2 1])
        res (ndarray/dot nda1 nda2)]
    (is (= [1 1] (mx-shape/->vec (shape res))))
    (is (= [11.0] (->vec res)))))

(deftest test-arrange
  (let [start 0
        stop 5
        step 0.5
        repeat 2]
    (is (= [0.0 0.0 0.5 0.5 1.0 1.0 1.5 1.5 2.0 2.0 2.5 2.5 3.0 3.0 3.5 3.5 4.0 4.0 4.5 4.5]
           (->vec (ndarray/arange start stop {:step step :repeat repeat}))))))

(deftest test->ndarray
  (let [nda1 (ndarray/->ndarray [5.0 -4.0])
        nda2 (ndarray/->ndarray [[1 2 3]
                                 [4 5 6]])
        nda3 (ndarray/->ndarray [[[7.0] [8.0]]])]
    (is (= [5.0 -4.0] (->vec nda1)))
    (is (= [2] (mx-shape/->vec (shape nda1))))
    (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (->vec nda2)))
    (is (= [2 3] (mx-shape/->vec (shape nda2))))
    (is (= [7.0 8.0] (->vec nda3)))
    (is (= [1 2 1] (mx-shape/->vec (shape nda3))))))

(deftest test-power
  (let [nda (ndarray/array [3 5] [2 1])]

    (let [nda-power-1 (ndarray/power 2 nda)]
      (is (= [2 1] (-> nda-power-1 shape mx-shape/->vec)))
      (is (= [8.0 32.0] (->vec nda-power-1))))

    (let [nda-power-2 (ndarray/power nda 2)]
      (is (= [2 1] (-> nda-power-2 shape mx-shape/->vec)))
      (is (= [9.0 25.0] (->vec nda-power-2))))

    (let [nda-power-3 (ndarray/power nda nda)]
      (is (= [2 1] (-> nda-power-3 shape mx-shape/->vec)))
      (is (= [27.0 3125.0] (->vec nda-power-3))))

    (let [nda-power-4 (ndarray/** nda 2)]
      (is (= [2 1] (-> nda-power-4 shape mx-shape/->vec)))
      (is (= [9.0 25.0] (->vec nda-power-4))))

    (let [nda-power-5 (ndarray/** nda nda)]
      (is (= [2 1] (-> nda-power-5 shape mx-shape/->vec)))
      (is (= [27.0 3125.0] (->vec nda-power-5))))

    (let [_  (ndarray/**= nda 2)]
      (is (= [2 1] (-> nda shape mx-shape/->vec)))
      (is (= [9.0 25.0] (->vec nda))))

    (let [_ (ndarray/set nda [3 5])
          _ (ndarray/**= nda nda)]
      (is (= [2 1] (-> nda shape mx-shape/->vec)))
      (is (= [27.0 3125.0] (->vec nda))))))

(deftest test-equal
  (let [nda1 (ndarray/array [1 2 3 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/equal nda1 nda2) shape mx-shape/->vec)))
    (is (= [1.0 0.0 1.0 0.0] (->vec (ndarray/equal nda1 nda2))))

    (is (= [2 2] (-> (ndarray/equal nda1 3) shape mx-shape/->vec)))
    (is (= [0.0 0.0 1.0 0.0] (->vec (ndarray/equal nda1 3))))))

(deftest test-not-equal
  (let [nda1 (ndarray/array [1 2 3 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/not-equal nda1 nda2) shape mx-shape/->vec)))
    (is (= [0.0 1.0 0.0 1.0] (->vec (ndarray/not-equal nda1 nda2))))

    (is (= [2 2] (-> (ndarray/not-equal nda1 3) shape mx-shape/->vec)))
    (is (= [1.0 1.0 0.0 1.0] (->vec (ndarray/not-equal nda1 3))))))

(deftest test-greater
  (let [nda1 (ndarray/array [1 2 4 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/> nda1 nda2) shape mx-shape/->vec)))
    (is (= [0.0 0.0 1.0 0.0] (->vec (ndarray/> nda1 nda2))))

    (is (= [2 2] (-> (ndarray/> nda1 2) shape mx-shape/->vec)))
    (is (= [0.0 0.0 1.0 1.0] (->vec (ndarray/> nda1 2))))))

(deftest test-greater-equal
  (let [nda1 (ndarray/array [1 2 4 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/>= nda1 nda2) shape mx-shape/->vec)))
    (is (= [1.0 0.0 1.0 0.0] (->vec (ndarray/>= nda1 nda2))))

    (is (= [2 2] (-> (ndarray/>= nda1 2) shape mx-shape/->vec)))
    (is (= [0.0 1.0 1.0 1.0] (->vec (ndarray/>= nda1 2))))))

(deftest test-lesser
  (let [nda1 (ndarray/array [1 2 4 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/< nda1 nda2) shape mx-shape/->vec)))
    (is (= [0.0 1.0 0.0 1.0] (->vec (ndarray/< nda1 nda2))))

    (is (= [2 2] (-> (ndarray/< nda1 2) shape mx-shape/->vec)))
    (is (= [1.0 0.0 0.0 0.0] (->vec (ndarray/< nda1 2))))))

(deftest test-lesser-equal
  (let [nda1 (ndarray/array [1 2 4 5] [2 2])
        nda2 (ndarray/array [1 4 3 6] [2 2])]

    (is (= [2 2] (-> (ndarray/<= nda1 nda2) shape mx-shape/->vec)))
    (is (= [1.0 1.0 0.0 1.0] (->vec (ndarray/<= nda1 nda2))))

    (is (= [2 2] (-> (ndarray/< nda1 2) shape mx-shape/->vec)))
    (is (= [1.0 1.0 0.0 0.0] (->vec (ndarray/<= nda1 2))))))

(deftest test-choose-element-0index
  (let [nda (ndarray/array [1 2 3 4 6 5] [2 3])
        indices (ndarray/array [0 1] [2])
        res (ndarray/choose-element-0index nda indices)]
    (is (= [1.0 6.0] (->vec res)))))

(deftest test-copy-to
  (let [source (ndarray/array [1 2 3] [1 3])
        dest (ndarray/empty [1 3])
        _ (ndarray/copy-to source dest)]
    (is (= [1 3] (-> dest shape mx-shape/->vec)))
    (is (= [1.0 2.0 3.0] (->vec dest)))))

(deftest test-abs
  (let [nda (ndarray/array [-1 -2 3] [3 1])]
    (is (= [1.0 2.0 3.0] (->vec (ndarray/abs nda))))))

(deftest test-sign
  (let [nda (ndarray/array [-1 -2 3] [3 1])]
    (is (= [-1.0 -1.0 1.0] (->vec (ndarray/sign nda))))))

(deftest test-round
  (let [nda (ndarray/array [1.5 2.1 3.7] [3 1])]
    (is (= [2.0 2.0 4.0] (->vec (ndarray/round nda))))))

(deftest test-ceil
  (let [nda (ndarray/array [1.5 2.1 3.7] [3 1])]
    (is (= [2.0 3.0 4.0] (->vec (ndarray/ceil nda))))))

(deftest test-floor
  (let [nda (ndarray/array [1.5 2.1 3.7] [3 1])]
    (is (= [1.0 2.0 3.0] (->vec (ndarray/floor nda))))))

(deftest test-square
  (let [nda (ndarray/array [1 2 3] [3 1])]
    (is (= [1.0 4.0 9.0] (->vec (ndarray/square nda))))))

(deftest test-exp
  (let [nda (ones [1])]
    (is (test-util/approx= 1e-3 2.71828 (ndarray/to-scalar (ndarray/exp nda))))))

(deftest test-log
  (let [nda (-> (ndarray/empty [1])
                (ndarray/set 10))]
    (is (test-util/approx= 1e-3 2.30258 (ndarray/to-scalar (ndarray/log nda))))))

(deftest test-cos
  (let [nda (-> (ndarray/empty [1])
                (ndarray/set 12))]
    (is (test-util/approx= 1e-3 0.8438539 (ndarray/to-scalar (ndarray/cos nda))))))

(deftest test-sin
  (let [nda (-> (ndarray/empty [1])
                (ndarray/set 12))]
    (is (test-util/approx= 1e-3 -0.536572918 (ndarray/to-scalar (ndarray/sin nda))))))

(deftest test-max
  (let [nda (ndarray/array [1.5 2.1 3.7] [3 1])]
    (is (test-util/approx= 1e-3 3.7 (ndarray/to-scalar (ndarray/max nda))))))

(deftest test-maximum
  (let [nda1 (ndarray/array [1.5 2.1 3.7] [3 1])
        nda2 (ndarray/array [4 1 3.5] [3 1])
        res (ndarray/maximum nda1 nda2)]
    (is (= [3 1] (-> res shape mx-shape/->vec)))
    (is (test-util/approx= 1e-3 [4.0 2.1 3.7] (->vec res)))))

(deftest test-min
  (let [nda (ndarray/array [1.5 2.1 3.7] [3 1])]
    (is (test-util/approx= 1e-3 1.5 (ndarray/to-scalar (ndarray/min nda))))))

(deftest test-minimum
  (let [nda1 (ndarray/array [1.5 2.1 3.7] [3 1])
        nda2 (ndarray/array [4 1 3.5] [3 1])
        res (ndarray/minimum nda1 nda2)]
    (is (= [3 1] (-> res shape mx-shape/->vec)))
    (is (test-util/approx= 1e-3 [1.5 1.0 3.5] (->vec res)))))

(deftest test-sum
  (let [nda (ndarray/array [1 2 3 4] [2 2])]
    (is (test-util/approx= 1e-3 10.0 (ndarray/to-scalar (ndarray/sum nda))))))

(deftest test-argmax-channel
  (let [nda (ndarray/array [1 2 4 3] [2 2])
        argmax (ndarray/argmax-channel nda)]
    (is (= [2] (-> argmax shape mx-shape/->vec)))
    (is (= [1.0 0.0] (->vec argmax)))))

(deftest test-concatenate-axis-0
  (let [nda1 (ndarray/array [1 2 4 3 3 3] [2 3])
        nda2 (ndarray/array [8 7 6] [1 3])
        res (ndarray/concatenate [nda1 nda2])]
    (is (= [3 3] (-> res shape mx-shape/->vec)))
    (is (= [1.0 2.0 4.0 3.0 3.0 3.0 8.0 7.0 6.0] (->vec res)))))

(deftest test-concatenate-axis-1
  (let [nda1 (ndarray/array [1 2 3 4] [2 2])
        nda2 (ndarray/array [5 6] [2 1])
        res (ndarray/concatenate [nda1 nda2] {:axis 1})]
    (is (= [2 3] (-> res shape mx-shape/->vec)))
    (is (= [1.0 2.0 5.0 3.0 4.0 6.0] (->vec res)))))

(deftest test-transpose
  (let [nda (ndarray/array [1 2 4 3 3 3] [2 3])]
    (is (= [1.0 2.0 4.0 3.0 3.0 3.0] (->vec nda)))
    (is (= [3 2] (-> (ndarray/t nda) shape mx-shape/->vec)))
    (is (= [1.0 3.0 2.0 3.0 4.0 3.0] (->vec (ndarray/t nda))))))

(def file-seq-num (atom 0))

(deftest test-save-and-load-with-names
  (let [filename (str (System/getProperty "java.io.tmpdir") "/ndarray" (swap! file-seq-num inc) ".bin")
        nda (ndarray/array [1 2 3] [3 1])
        _ (ndarray/save filename {"local" nda})
        load-map (ndarray/load filename)]
    (is (= ["local"] (keys load-map)))
    (is (= 1 (count (vals load-map))))
    (is (= [3 1] (-> (get load-map "local") shape mx-shape/->vec)))
    (is (= [1.0 2.0 3.0] (->vec (get load-map "local"))))))

(deftest test-save-to-file-and-load-from-file
  (let [filename (str (System/getProperty "java.io.tmpdir") "/ndarray" (swap! file-seq-num inc) ".bin")
        nda (ndarray/array [1 2 3] [3 1])
        _ (ndarray/save-to-file filename nda)
        load-nda (ndarray/load-from-file filename)]
    (is (= [3 1] (-> load-nda shape mx-shape/->vec)))
    (is (= [1.0 2.0 3.0] (->vec load-nda)))))

(deftest test-get-context
  (let [nda (ones [3 2])
        ctx (ndarray/context nda)]
    (is (= "cpu" (ctx/device-type ctx)))
    (is (= 0 (ctx/device-id ctx)))))

(deftest test-equals
  (let [nda1 (ndarray/array [1 2 3] [3 1])
        nda2 (ndarray/array [1 2 3] [3 1])
        nda3 (ndarray/array [1 2 3] [1 3])
        nda4 (ndarray/array [3 2 3] [3 1])]
    (is (= nda1 nda2))
    (is (not= nda1 nda3))
    (is (not= nda1 nda4))))

(deftest test-slice
  (let [nda (ndarray/array [1 2 3 4 5 6] [3 2])]

    (let [nda1 (ndarray/slice nda 1)]
      (is (= [1 2] (-> nda1 shape mx-shape/->vec)))
      (is (= [3.0 4.0] (->vec nda1))))

    (let [nda2 (ndarray/slice nda 1 3)]
      (is (= [2 2] (-> nda2 shape mx-shape/->vec)))
      (is (= [3.0 4.0 5.0 6.0] (->vec nda2))))))

(deftest test-at
  (let [nda (ndarray/array [1 2 3 4 5 6] [3 2])
        res (ndarray/at nda 1)]
    (is (= [2] (-> res shape mx-shape/->vec)))
    (is (= [3 4] (-> res ndarray/->int-vec)))))

(deftest test-reshape
  (let [nda (ndarray/array [1 2 3 4 5 6] [3 2])
        nda1 (ndarray/reshape nda [2 3])]
    (is (= [2 3] (-> nda1 shape mx-shape/->vec)))
    (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (->vec nda1)))))

(deftest test-dispose-deps
  (let [nda1 (ones [1 2])
        nda2 (ones [1 2])
        nda3 (ones [1 2])
        nda-with-deps (ndarray/+ nda3 (ndarray/+ nda1 nda2))]
    (is (= 4 (ndarray/size (ndarray/dependencies nda-with-deps))))
    (is (contains? (-> (ndarray/dependencies nda-with-deps) keys set) (ndarray/handle nda1)))
    (is (contains? (-> (ndarray/dependencies nda-with-deps) keys set) (ndarray/handle nda2)))
    (is (contains? (-> (ndarray/dependencies nda-with-deps) keys set) (ndarray/handle nda3)))
    (is (not (ndarray/is-disposed nda1)))
    (is (not (ndarray/is-disposed nda2)))
    (is (not (ndarray/is-disposed nda3)))

    (let [nda-no-deps (ndarray/dispose-deps nda-with-deps)]
      (is (= 0 (ndarray/size (ndarray/dependencies nda-no-deps))))
      (is (ndarray/is-disposed nda1))
      (is (ndarray/is-disposed nda2))
      (is (ndarray/is-disposed nda3)))))

(deftest test-dispose-deps-except
  (let [nda1 (ones [1 2])
        nda2 (ones [1 2])
        nda3 (ones [1 2])
        nda1-2 (ndarray/+ nda1 nda2)]

    (let [res (-> (ndarray/+ nda1 nda2)
                  (ndarray/+ nda1-2)
                  (ndarray/+ nda3)
                  (ndarray/dispose-deps-except nda1-2))]
      (is (= 3 (ndarray/size (ndarray/dependencies res))))
      (is (contains? (-> (ndarray/dependencies res) keys set) (ndarray/handle nda1)))
      (is (contains? (-> (ndarray/dependencies res) keys set) (ndarray/handle nda2)))
      (is (contains? (-> (ndarray/dependencies res) keys set) (ndarray/handle nda1-2)))
      (is (not (ndarray/is-disposed nda1)))
      (is (not (ndarray/is-disposed nda2)))
      (is (ndarray/is-disposed nda3)))))

(deftest test-serialize-deserialize
  (let [nda (ndarray/* (ndarray/ones [1 2]) 3)
        nda-bytes (ndarray/serialize nda)
        nda-copy (ndarray/deserialize nda-bytes)]
    (is (= nda nda-copy))))

(deftest test-dtype-int32
  (let [nda (ndarray/* (ones [1 2] {:dtype dtype/INT32}) 2)]
    (is (= dtype/INT32 (ndarray/dtype nda)))
    (is (= 8 (count (ndarray/->raw nda))))
    (is (= [2.0 2.0] (ndarray/->float-vec nda)))
    (is (= [2 2] (ndarray/->int-vec nda)))
    (is (= [2.0 2.0] (ndarray/->double-vec nda)))
    (is (= [(byte 2) (byte 2)] (ndarray/->byte-vec nda)))))

(deftest test-dtype-uint8
  (let [nda (ndarray/* (ones [1 2] {:dtype dtype/UINT8}) 2)]
    (is (= dtype/UINT8 (ndarray/dtype nda)))
    (is (= 2 (count (ndarray/->raw nda))))
    (is (= [2.0 2.0] (ndarray/->float-vec nda)))
    (is (= [2 2] (ndarray/->int-vec nda)))
    (is (= [2.0 2.0] (ndarray/->double-vec nda)))
    (is (= [(byte 2) (byte 2)] (ndarray/->byte-vec nda)))))

(deftest test-dtype-float64
  (let [nda (ndarray/* (ones [1 2] {:dtype dtype/FLOAT64}) 2)]
    (is (= dtype/FLOAT64 (ndarray/dtype nda)))
    (is (= 16 (count (ndarray/->raw nda))))
    (is (= [2.0 2.0] (ndarray/->float-vec nda)))
    (is (= [2 2] (ndarray/->int-vec nda)))
    (is (= [2.0 2.0] (ndarray/->double-vec nda)))
    (is (= [(byte 2) (byte 2)] (ndarray/->byte-vec nda)))))

(deftest test->nd-vec
  (is (= [[[1.0]]]
         (ndarray/->nd-vec (ndarray/array [1] [1 1 1]))))
  (is (= [[[1.0]] [[2.0]] [[3.0]]]
         (ndarray/->nd-vec (ndarray/array [1 2 3] [3 1 1]))))
  (is (= [[[1.0 2.0]] [[3.0 4.0]] [[5.0 6.0]]]
         (ndarray/->nd-vec (ndarray/array [1 2 3 4 5 6] [3 1 2]))))
  (is (= [[[1.0] [2.0]] [[3.0] [4.0]] [[5.0] [6.0]]]
         (ndarray/->nd-vec (ndarray/array [1 2 3 4 5 6] [3 2 1]))))
  (is (thrown-with-msg? Exception #"Invalid input array"
                         (ndarray/->nd-vec [1 2 3 4 5]))))
