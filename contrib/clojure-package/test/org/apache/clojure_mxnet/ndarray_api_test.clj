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

(ns org.apache.clojure-mxnet.ndarray-api-test
  (:require [org.apache.clojure-mxnet.base :as base]
            [org.apache.clojure-mxnet.context :as ctx]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.ndarray :as ndarray :refer [->vec zeros ones += -= *= full shape shape-vec]]
            [org.apache.clojure-mxnet.ndarray-api :as ndarray-api]
            [org.apache.clojure-mxnet.shape :as mx-shape :refer [->shape]]
            [org.apache.clojure-mxnet.test-util :as test-util :refer [approx=]]
            [org.apache.clojure-mxnet.util :as util :refer [->option]]
            [clojure.test :refer :all]))

(deftest test-activation
  (let [data (ndarray/array [2 1 0 -1 -2] [1 5])
        relu (ndarray-api/activation data "relu")
        sigmoid (ndarray-api/activation data "sigmoid")
        softsign (ndarray-api/activation data "softsign")
        out (ndarray/zeros [1 5])
        _ (ndarray-api/activation {:data data :act-type "relu" :out out})]
    (is (= [2.0 1.0 0.0 0.0 0.0] (->vec relu)))
    (is (approx= 1e-3 [0.881 0.731 0.5 0.269 0.119] (->vec sigmoid)))
    (is (approx= 1e-3 [0.666 0.5 0.0 -0.5 -0.666] (->vec softsign)))
    (is (= [2.0 1.0 0.0 0.0 0.0] (->vec out)))))

(deftest test-bilinear-sampler
  (let [data (ndarray/array [1 4 3 6
                             1 8 8 9
                             0 4 1 5
                             1 0 1 3]
                            [1 1 4 4])
        affine (ndarray/array [2 0 0
                               0 2 0]
                              [1 6])
        grid (ndarray-api/grid-generator {:data affine :transform-type "affine" :target-shape [4 4]})
        out (ndarray-api/bilinear-sampler data grid)]
    (is (approx= 1e-3
                 [0.0 0.0 0.0 0.0
                  0.0 3.5 6.5 0.0
                  0.0 1.25 2.5 0.0
                  0.0 0.0 0.0 0.0]
                 (->vec out)))))

(deftest test-cast
  (let [nda1 (ndarray/array [0.9 1.3] [2])
        nda2 (ndarray/array [1e20 11.1] [2])
        nda3 (ndarray/array [300 11.1 10.9 -1 -3] [5])
        out (ndarray/zeros [2] {:dtype dtype/INT32})
        _ (ndarray-api/cast {:data nda1 :dtype (str dtype/INT32) :out out})]
    (is (= [0.0 1.0] (->vec (ndarray-api/cast nda1 (str dtype/INT32)))))
    (is (= [(float 1e20) (float 11.1)] (->vec (ndarray-api/cast nda2 (str dtype/FLOAT32)))))
    ;; uint8 gets converted to native types after ->vec
    (is (= [44.0 11.0 10.0 -1.0 -3.0] (->vec (ndarray-api/cast nda3 "uint8"))))))

(deftest test-concat
  (let [nda1 (ndarray/zeros [1 2])
        nda2 (ndarray/ones [1 2])
        out (ndarray/zeros [1 4])
        res1 (ndarray-api/concat [nda1 nda2] 2) ;; num_args=2, dim=1 (default)
        res2 (ndarray-api/concat {:data [nda1 nda2] :num-args 2 :dim 0}) ;; num_args=2, dim=0
        res3 (ndarray-api/concat {:data [nda1 nda2 nda1] :num-args 3 :dim 1}) ;; num_args=3, dim=1
        _ (ndarray-api/concat {:data [nda1 nda2] :num-args 2 :dim 1 :out out}) ;; store result in out
        ]
    (is (= [0.0 0.0 1.0 1.0] (->vec res1)))
    (is (= [1 4] (shape-vec res1)))
    (is (= [0.0 0.0 1.0 1.0] (->vec res2)))
    (is (= [2 2] (shape-vec res2)))
    (is (= [0.0 0.0 1.0 1.0 0.0 0.0] (->vec res3)))
    (is (= [1 6] (shape-vec res3)))
    (is (= [0.0 0.0 1.0 1.0] (->vec out)))
    (is (= [1 4] (shape-vec out)))))

(deftest test-embedding
  (let [input-dim 4
        output-dim 5
        w (ndarray/array [0.  1.  2.  3.  4.
                          5.  6.  7.  8.  9.
                          10. 11. 12. 13. 14.
                          15. 16. 17. 18. 19.]
                         [4 5])
        x (ndarray/array [1. 3.
                          0. 2.]
                         [2 2])
        out (ndarray-api/embedding x w input-dim output-dim)]
    (is (= [5.  6.  7.  8.  9.
            15. 16. 17. 18. 19.
            0.  1.  2.  3.  4.
            10. 11. 12. 13. 14.]
           (->vec out)))
    (is (= [2 2 5] (shape-vec out)))))

(deftest test-flatten
  (let [nda (ndarray/array [1 2 3
                            4 5 6
                            7 8 9
                            1 2 3
                            4 5 6
                            7 8 9]
                           [2 3 3])
        out (ndarray/zeros [2 9])
        res (ndarray-api/flatten {:data nda})
        _ (ndarray-api/flatten {:data nda :out out})]
    (is (= [1. 2. 3. 4. 5. 6. 7. 8. 9.
            1. 2. 3. 4. 5. 6. 7. 8. 9.] (->vec res)))
    (is (= [2 9] (shape-vec res)))
    (is (= [1. 2. 3. 4. 5. 6. 7. 8. 9.
            1. 2. 3. 4. 5. 6. 7. 8. 9.] (->vec out)))
    (is (= [2 9] (shape-vec out)))))

(deftest test-instance-norm
  (let [x (ndarray/array [1.1 2.2 3.3 4.4] [2 1 2])
        gamma (ndarray/array [1.5] [1])
        beta (ndarray/array [0.5] [1])
        res (ndarray-api/instance-norm x gamma beta)]
    (is (approx= 1e-4 [-0.9975 1.9975
                       -0.9975 1.9975] (->vec res)))
    (is (= [2 1 2] (shape-vec res)))))

(deftest test-l2-normalization
  (let [x (ndarray/array [1 2 3 4 2 2 5 6] [2 2 2])
        res1 (ndarray-api/l2-normalization {:data x}) ;; instance-wise
        res2 (ndarray-api/l2-normalization {:data x :mode "instance"})
        res3 (ndarray-api/l2-normalization {:data x :mode "channel"})
        res4 (ndarray-api/l2-normalization {:data x :mode "spatial"})]
    (is (approx= 1e-4 [0.1825 0.3651
                       0.5477 0.7303
                       0.2407 0.2407
                       0.6019 0.7223] (->vec res1)))
    (is (approx= 1e-4 [0.1825 0.3651
                       0.5477 0.7303
                       0.2407 0.2407
                       0.6019 0.7223] (->vec res2)))
    (is (approx= 1e-4 [0.3162 0.4472
                       0.9486 0.8944
                       0.3714 0.3162
                       0.9284 0.9486] (->vec res3)))
    (is (approx= 1e-4 [0.4472 0.8944
                       0.6    0.8
                       0.7071 0.7071
                       0.6402 0.7682] (->vec res4)))))

(deftest test-pad
  (let [x (ndarray/array [1 2 3
                          4 5 6
                          7 8 9
                          10 11 12
                          11 12 13
                          14 15 16
                          17 18 19
                          20 21 22]
                         [2 2 2 3])
        res1 (ndarray-api/pad x "edge" [0,0,0,0,1,1,1,1])
        res2 (ndarray-api/pad {:data x :mode "constant" :pad-width [0,0,0,0,1,1,1,1] :constant-value 0})]
    (is (= [1.   1.   2.   3.   3.
            1.   1.   2.   3.   3.
            4.   4.   5.   6.   6.
            4.   4.   5.   6.   6.
            7.   7.   8.   9.   9.
            7.   7.   8.   9.   9.
            10.  10.  11.  12.  12.
            10.  10.  11.  12.  12.
            11.  11.  12.  13.  13.
            11.  11.  12.  13.  13.
            14.  14.  15.  16.  16.
            14.  14.  15.  16.  16.
            17.  17.  18.  19.  19.
            17.  17.  18.  19.  19.
            20.  20.  21.  22.  22.
            20.  20.  21.  22.  22.] (->vec res1)))
    (is (= [2 2 4 5] (shape-vec res1)))
    (is (= [0.   0.   0.   0.   0.
            0.   1.   2.   3.   0.
            0.   4.   5.   6.   0.
            0.   0.   0.   0.   0.
            
            0.   0.   0.   0.   0.
            0.   7.   8.   9.   0.
            0.  10.  11.  12.   0.
            0.   0.   0.   0.   0.
            
            0.   0.   0.   0.   0.
            0.  11.  12.  13.   0.
            0.  14.  15.  16.   0.
            0.   0.   0.   0.   0.
            
            0.   0.   0.   0.   0.
            0.  17.  18.  19.   0.
            0.  20.  21.  22.   0.
            0.   0.   0.   0.   0.] (->vec res2)))
    (is (= [2 2 4 5] (shape-vec res2)))))

(deftest test-roi-pooling
  (let [xi [[[[  0.,   1.,   2.,   3.,   4.,   5.],
              [  6.,   7.,   8.,   9.,  10.,  11.],
              [ 12.,  13.,  14.,  15.,  16.,  17.],
              [ 18.,  19.,  20.,  21.,  22.,  23.],
              [ 24.,  25.,  26.,  27.,  28.,  29.],
              [ 30.,  31.,  32.,  33.,  34.,  35.],
              [ 36.,  37.,  38.,  39.,  40.,  41.],
              [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
        x (ndarray/array (-> xi flatten vec) [1 1 8 6])
        y (ndarray/array [0 0 0 4 4] [1 5])
        res1 (ndarray-api/roi-pooling x y [2 2] 1.0)
        res2 (ndarray-api/roi-pooling x y [2 2] 0.7)]
    (is (= [14. 16. 26. 28.] (->vec res1)))
    (is (= [1 1 2 2] (shape-vec res1)))
    (is (= [7. 9. 19. 21.] (->vec res2)))
    (is (= [1 1 2 2] (shape-vec res2)))))

(deftest test-reshape
  (let [x (ndarray/array (vec (range 4)) [4])
        y (ndarray/array (vec (range 24)) [2 3 4])
        z (ndarray/array (vec (range 120)) [2 3 4 5])
        res1 (ndarray-api/reshape {:data x :shape [2 2]})]
    (is (= [0. 1. 2. 3.] (->vec res1)))
    (is (= [2 2] (shape-vec res1)))
    (is (= (map float (range 24)) (->vec (ndarray-api/reshape {:data y :shape [4 0 2]}))))
    (is (= [4 3 2] (shape-vec (ndarray-api/reshape {:data y :shape [4 0 2]}))))
    (is (= [2 3 4] (shape-vec (ndarray-api/reshape {:data y :shape [2 0 0]}))))
    (is (= [6 1 4] (shape-vec (ndarray-api/reshape {:data y :shape [6 1 -1]}))))
    (is (= [3 1 8] (shape-vec (ndarray-api/reshape {:data y :shape [3 -1 8]}))))
    (is (= [24] (shape-vec (ndarray-api/reshape {:data y :shape [-1]}))))
    (is (= [2 3 4] (shape-vec (ndarray-api/reshape {:data y :shape [-2]}))))
    (is (= [2 3 4] (shape-vec (ndarray-api/reshape {:data y :shape [2 -2]}))))
    (is (= [2 3 4 1 1] (shape-vec (ndarray-api/reshape {:data y :shape [-2 1 1]}))))
    (is (= [6 4] (shape-vec (ndarray-api/reshape {:data y :shape [-3 4]}))))
    (is (= [6 20] (shape-vec (ndarray-api/reshape {:data z :shape [-3 -3]}))))
    (is (= [2 12] (shape-vec (ndarray-api/reshape {:data y :shape [0 -3]}))))
    (is (= [6 4] (shape-vec (ndarray-api/reshape {:data y :shape [-3 -2]}))))
    (is (= [1 2 3 4] (shape-vec (ndarray-api/reshape {:data y :shape [-4 1 2 -2]}))))
    (is (= [2 1 3 4] (shape-vec (ndarray-api/reshape {:data y :shape [2 -4 -1 3 -2]}))))))

(deftest test-sequence-last
  (let [xi [[[  1.,   2.,   3.],
             [  4.,   5.,   6.],
             [  7.,   8.,   9.]],
            
            [[ 10.,   11.,   12.],
             [ 13.,   14.,   15.],
             [ 16.,   17.,   18.]],
            
            [[  19.,   20.,   21.],
             [  22.,   23.,   24.],
             [  25.,   26.,   27.]]]
        x (ndarray/array (-> xi flatten vec) [3 3 3])
        seq-len1 (ndarray/array [1 1 1] [3])
        seq-len2 (ndarray/array [1 2 3] [3])
        ;; This test is failing with an exception
        ;; (most likely a scala generation issue)
        ;; res1 (ndarray-api/sequence-last x nil)
        ]
    ;; (is (= [] (->vec res1)))
))

(deftest test-sequence-mask
  (let [xi [[[  1.,   2.,   3.],
             [  4.,   5.,   6.]],
            
            [[  7.,   8.,   9.],
             [ 10.,  11.,  12.]],
            
            [[ 13.,  14.,   15.],
             [ 16.,  17.,   18.]]]
        x (ndarray/array (-> xi flatten vec) [3 2 3])
        seq-len1 (ndarray/array [1 1] [2])
        seq-len2 (ndarray/array [2 3] [2])
        ;; Same issue as previous test
        ;; res1 (ndarray-api/sequence-mask x seq-len1)
        ]
    ;; (is (= [] (->vec res1)))
))

(deftest test-slice-channel
  (let [xi [[[ 1.] [ 2.]]
            [[ 3.] [ 4.]]
            [[ 5.] [ 6.]]]
        x (ndarray/array (-> xi flatten vec) [3 2 1])
        res1 (ndarray-api/slice-channel {:data x :num-outputs 2 :axis 1})
        res2 (ndarray-api/slice-channel {:data x :num-outputs 3 :axis 0})
        res3 (ndarray-api/slice-channel {:data x :num-outputs 3 :axis 0 :squeeze-axis 1})]
    (is (= [1. 3. 5.] (->vec res1)))
    (is (= [3 1 1] (shape-vec res1)))
    (is (= [1. 2.] (->vec res2)))
    (is (= [1 2 1] (shape-vec res2)))
    (is (= [1. 2.] (->vec res3)))
    (is (= [2 1] (shape-vec res3)))))

(deftest test-softmax-activation
  (let [x (ndarray/array [1 1 1 1 1 1] [2 3])
        res1 (ndarray-api/softmax-activation {:data x :mode "instance"})]
    (is (approx= 1e-3 [0.333 0.333 0.333
                       0.333 0.333 0.333] (->vec res1)))
    (is (= [2 3] (shape-vec res1)))))

(deftest test-softmax-output
  (let [datai [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
        data (ndarray/array (-> datai flatten vec) [4 4])
        label (ndarray/array [1,0,2,3] [4])
        res1 (ndarray-api/softmax-output data label)]
    (is (approx= 1e-4 [0.0321 0.0871 0.2369 0.6439
                       0.25 0.25 0.25 0.25
                       0.25 0.25 0.25 0.25
                       0.25 0.25 0.25 0.25] (->vec res1)))
    (is (= [4 4] (shape-vec res1)))))

(deftest test-swap-axis
  (let [x (ndarray/array (range 3) [1 3])
        y (ndarray/array (range 8) [2 2 2])
        res1 (ndarray-api/swap-axis {:data x :dim1 0 :dim2 1})
        res2 (ndarray-api/swap-axis {:data y :dim1 0 :dim2 2})]
    (is (= [0. 1. 2.] (->vec res1)))
    (is (= [3 1] (shape-vec res1)))
    (is (= [0. 4. 2. 6. 1. 5. 3. 7.] (->vec res2)))
    (is (= [2 2 2] (shape-vec res2)))))

(deftest test-abs
  (let [x (ndarray/array [-2 0 3] [3])
        res1 (ndarray-api/abs {:data x})]
    (is (= [2. 0. 3.] (->vec res1)))
    (is (= [3] (shape-vec res1)))))

(deftest test-arccos
  (let [x (ndarray/array [-1 -0.707 0 0.707 1] [5])
        pi Math/PI
        res1 (ndarray-api/arccos {:data x})]
    (is (approx= 1e-3 [pi (* 0.75 pi) (* 0.5 pi) (* 0.25 pi) 0.] (->vec res1)))))

(deftest test-arcsin
  (let [x (ndarray/array [-1 -0.707 0 0.707 1] [5])
        pi Math/PI
        res1 (ndarray-api/arcsin {:data x})]
    (is (approx= 1e-3 [(- (* 0.5 pi)) (- (* 0.25 pi)) 0 (* 0.25 pi) (* 0.5 pi)] (->vec res1)))))

(deftest test-argmax
  (let [x (ndarray/array (range 6) [2 3])
        res1 (ndarray-api/argmax {:data x :axis 0})
        res2 (ndarray-api/argmax {:data x :axis 1})
        res3 (ndarray-api/argmax {:data x :axis 0 :keepdims true})
        res4 (ndarray-api/argmax {:data x :axis 1 :keepdims true})]
    (is (= [1. 1. 1.] (->vec res1)))
    (is (= [3] (shape-vec res1)))
    (is (= [2. 2.] (->vec res2)))
    (is (= [2] (shape-vec res2)))
    (is (= [1. 1. 1.] (->vec res3)))
    (is (= [1 3] (shape-vec res3)))
    (is (= [2. 2.] (->vec res4)))
    (is (= [2 1] (shape-vec res4)))))

(deftest test-argmax-channel
  (let [x (ndarray/array (range 6) [2 3])
        res1 (ndarray-api/argmax-channel {:data x})]
    (is (= [2. 2.] (->vec res1)))
    (is (= [2] (shape-vec res1)))))

(deftest test-argmin
  (let [x (ndarray/array (reverse (range 6)) [2 3])
        res1 (ndarray-api/argmin {:data x :axis 0})
        res2 (ndarray-api/argmin {:data x :axis 1})
        res3 (ndarray-api/argmin {:data x :axis 0 :keepdims true})
        res4 (ndarray-api/argmin {:data x :axis 1 :keepdims true})]
    (is (= [1. 1. 1.] (->vec res1)))
    (is (= [3] (shape-vec res1)))
    (is (= [2. 2.] (->vec res2)))
    (is (= [2] (shape-vec res2)))
    (is (= [1. 1. 1.] (->vec res3)))
    (is (= [1 3] (shape-vec res3)))
    (is (= [2. 2.] (->vec res4)))
    (is (= [2 1] (shape-vec res4)))))

(deftest test-argsort
  (let [x (ndarray/array [0.3  0.2  0.4
                          0.1  0.3  0.2]
                         [2 3])
        y (ndarray/array [0.3 0.2 0.4 0.1 0.3 0.2] [6])
        res1 (ndarray-api/argsort {:data x})
        res2 (ndarray-api/argsort {:data x :axis 0})
        res3 (ndarray-api/argsort {:data y})]
    (is (= [1. 0. 2.
            0. 2. 1.]
           (->vec res1)))
    (is (= [2 3] (shape-vec res1)))
    (is (= [1. 0. 1.
            0. 1. 0.]
           (->vec res2)))
    (is (= [2 3] (shape-vec res1)))
    (is (= [3. 1. 5. 0. 4. 2.] (->vec res3)))
    (is (= [6] (shape-vec res3)))))

(deftest test-batch-take
  (let [x (ndarray/array (range 6) [3 2])
        i (ndarray/as-type (ndarray/array [0 1 0] [3]) dtype/INT32)
        res1 (ndarray-api/batch-take x i)        ]
    (is (= [0. 3. 4.] (->vec res1)))))

(deftest test-broadcast-add
  (let [x (ndarray/ones [2 3])
        y (ndarray/array (range 2) [2 1])
        res1 (ndarray-api/broadcast-add x y)]
    (is (= [1. 1. 1. 2. 2. 2.] (->vec res1)))
    (is (= [2 3] (shape-vec res1)))))
