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

(ns org.apache.clojure-mxnet.operator-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.test-util :as test-util]
            [clojure.test :refer :all])
  (:import (org.apache.mxnet NDArray)))

(defn approx= [tolerance x y]
  (test-util/approx= tolerance
                     (if (instance? NDArray x) (ndarray/->vec x) x)
                     (if (instance? NDArray y) (ndarray/->vec y) y)))

(deftest test-elementwise-sum
  (let [n 4
        shape-vec [5 5 3]
        inputs (mapv (fn [i] (sym/variable (str "arg" i))) (range n))
        out (sym/element-wise-sum "esum" inputs)
        arr (into [] (repeatedly n #(random/uniform -10 10 shape-vec)))
        arr-grad (into [] (repeatedly n #(ndarray/empty shape-vec)))
        exec (sym/bind out (context/default-context) arr arr-grad)
        forward-output (-> exec (executor/forward) (executor/outputs) first)
        forward-output-expected (reduce sym/+ arr)]
    (approx= 1e-4 forward-output-expected forward-output)

    ;; backward
    (let [out-grad (random/uniform -10 10 shape-vec)
          _ (executor/backward exec out-grad)]
      (doseq [grad arr-grad]
        (is (= out-grad grad))))))

(deftest test-concat
  (let [shape-vecs [[2 2] [3 2]]
        x (sym/variable "x")
        y (sym/variable "y")
        out (sym/concat "conc" nil [x y] {:dim 0})
        arr (mapv #(ndarray/empty %) shape-vecs)
        arr-np (mapv #(ndarray/copy %) arr)
        arr-grad (map #(ndarray/empty %) shape-vecs)
        arg-names (sym/list-arguments out)
        grad-map (zipmap arg-names arr-grad)
        args (sym/list-arguments out)
        [arg-shapes out-shapes aux-shapes] (sym/infer-shape out (zipmap args shape-vecs))
        out-shape-vec (first out-shapes)
        out-grad (ndarray/empty out-shape-vec)
        exec1 (sym/bind out (context/default-context) arr grad-map)
        out1 (-> (executor/forward exec1)
                 (executor/outputs)
                 (first))
        ret (ndarray/concatenate arr)]
    (is (= out1 ret))

    ;;backward
    (ndarray/copy-to out1 out-grad)
    (ndarray/+= out-grad 1)
    (executor/backward exec1 out-grad)
    (let [grads arr-grad
          np-grads arr-np]
      (is (= grads (mapv #(ndarray/+ % 1) np-grads))))))

(defn check-regression [model forward-fn backward-fn]
  (let [shape-vec [3 1]
        arr-data (random/uniform -1 1 shape-vec)
        arr-label (random/uniform -1 1 [(first shape-vec)])
        arr-grad (ndarray/empty shape-vec)
        exec1 (sym/bind model (context/default-context) [arr-data arr-label] {:data arr-grad})
        out1 (-> exec1 (executor/forward) (executor/outputs) first)
        np-out (map forward-fn
                    (ndarray/->vec arr-data))]
    (is (= shape-vec (-> out1 ndarray/shape mx-shape/->vec)))
    (is (approx= 1e-6 np-out out1))

    ;;backward
    (executor/backward exec1)
    (let [npout-back (mapv backward-fn
                           np-out (ndarray/->vec arr-label))]
      (is (approx= 1e-6 npout-back arr-grad)))))

(deftest test-regression
  (check-regression (sym/logistic-regression-output {:data (sym/variable "data") :label (sym/variable "label")})
                    (fn [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))
                    (fn [x y] (- x y)))
  (check-regression (sym/linear-regression-output {:data (sym/variable "data") :label (sym/variable "label")})
                    (fn [x] x)
                    (fn [x y] (- x y))))

(deftest swap-axes
  (let [data (sym/variable "data")
        shape-vec [2 3 4]
        arr-data (ndarray/ones shape-vec)]

    (->  (ndarray/slice arr-data 0)
         (ndarray/set 1))

    (->  (ndarray/slice arr-data 1)
         (ndarray/set 2))

    ;;  [[[ 1.,  1.,  1.,  1.],
    ;;    [ 1.,  1.,  1.,  1.],
    ;;    [ 1.,  1.,  1.,  1.]],
    ;;
    ;;  [[ 2.,  2.,  2.,  2.],
    ;;   [ 2.,  2.,  2.,  2.],
    ;;   [ 2.,  2.,  2.,  2.]]]

    (let [swap0 (sym/swap-axis {:data data :dim1 0 :dim2 2})
          swap (sym/swap-axis {:data swap0 :dim1 1 :dim2 2})
          exec (sym/bind swap (context/default-context) arr-data)
          out (-> (executor/forward exec)
                  (executor/outputs)
                  first)]
     ;;  After swapaxes(swapaxes(arrData, 0, 2), 1, 2)
     ;;   out should be
     ;;    [[[ 1.,  1.,  1.],
     ;;       [ 2.,  2.,  2.]],
     ;;
     ;;      [[ 1.,  1.,  1.],
     ;;       [ 2.,  2.,  2.]],
     ;;
     ;;      [[ 1.,  1.,  1.],
     ;;       [ 2.,  2.,  2.]],
     ;;
     ;;      [[ 1.,  1.,  1.],
     ;;       [ 2.,  2.,  2.]]]
      (= [4 2 3] (mx-shape/->vec (ndarray/shape out)))
      (doseq [i (range 4)]
        (let [val (ndarray/->vec (ndarray/slice out i))]
          (is (approx= 1e-6 [1 1 1 2 2 2] val)))))))

(defn check-symbolic-forward [test-sym location expected tolerance]
  (let [arr-data (mapv #(ndarray/copy %) location)
        arr-grad (mapv #(ndarray/empty (mx-shape/->vec (ndarray/shape %))) location)
        exec (sym/bind test-sym (context/default-context) arr-data arr-grad)
        outputs (-> exec
                    (executor/forward)
                    (executor/outputs))]
    (is (every? true? (map
                       (fn [x y]
                         #_(println "expected " (ndarray/->vec x))
                         #_(println "actual " (ndarray/->vec y))
                         (approx= tolerance x y))
                       expected
                       outputs)))))

(defn check-symbolic-backward [test-sym location grad expected tolerance]
  (let [arr-data (mapv #(ndarray/copy %) location)
        arr-grad (mapv #(ndarray/empty (mx-shape/->vec (ndarray/shape %))) location)
        out-grad (mapv #(ndarray/copy %) grad)
        exec (sym/bind test-sym (context/default-context) arr-data arr-grad)
        exec (-> exec
                 (executor/forward)
                 (executor/backward out-grad))
        grad-arrays (executor/grad-arrays exec)]
    (is (every? true? (map
                       (fn [x y]
                         #_(println "expected " (ndarray/->vec x))
                         #_(println "actual " (ndarray/->vec y))
                         (approx= tolerance x y))
                       expected
                       grad-arrays)))))

(deftest test-scalar-op
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5)
        ;; (4x + 2)/2
        test (-> (sym/* data 4)
                 (sym/+ 2)
                 (sym/div 2))
        npout (-> (ndarray/* data-tmp 4)
                  (ndarray/+ 2)
                  (ndarray/div 2))
        ;; backward deriv is 2
        np-out-grad (ndarray/* (ndarray/ones shape-vec) 2)]

    (check-symbolic-forward test [data-tmp] [npout] 1e-5)
    (check-symbolic-backward test [data-tmp] [(ndarray/ones shape-vec)] [np-out-grad] 1e-5)))

(deftest ones
  (let [ones (sym/ones [2 2])
        exec (sym/simple-bind ones (context/default-context))]
    (is (approx= 1e-4
                 [1 1 1 1]
                 (-> exec (executor/forward) (executor/outputs) (first))))))

(deftest zeros
  (let [zeros (sym/zeros [2 2])
        exec (sym/simple-bind zeros (context/default-context))]
    (is (approx= 1e-4
                 [0 0 0 0]
                 (-> exec (executor/forward) (executor/outputs) (first))))))

(deftest test-arange
  (let [start 1
        stop 100
        step 2
        result (range start stop step)
        x (sym/arange start stop {:step step})
        exec (sym/simple-bind x (context/default-context))]
    (executor/forward exec)
    (is (= 0 (count (executor/grad-arrays exec))))
    (is (approx= 1e-4 result (-> (executor/outputs exec) (first))))))

(deftest test-arange-with-inference
  (let [arange (sym/arange-with-inference 0)
        data (sym/variable "data")
        added (sym/+ arange data)
        result (range 0 4)
        data-tmp (ndarray/zeros [4])
        exec (sym/bind added (context/default-context) {"data" data-tmp})]
    (executor/forward exec)
    (is (= 0 (count (executor/grad-arrays exec))))
    (is (approx= 1e-4 result (-> (executor/outputs exec) (first))))))

(deftest test-scalar-pow
  (let [data (sym/variable "data")
        shape-vec [1 1]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 3)
        data-tmp-powered (ndarray/* (ndarray/ones shape-vec) 9)
        test (sym/** data 2)]
    (check-symbolic-forward test [data-tmp] [data-tmp-powered] 1e-5)
    (check-symbolic-backward test [data-tmp] [(ndarray/ones shape-vec)] [(ndarray/* data-tmp 2)] 1e-5)))

(deftest test-symbol-pow
  (let [shape-vec [1 1]
        data (sym/variable "data")
        data-tmp (ndarray/* (ndarray/ones shape-vec) 2)
        exp (sym/variable "exp")
        exp-tmp (ndarray/* (ndarray/ones shape-vec) 3)
        test (sym/** data exp)]
    (check-symbolic-forward test [data-tmp exp-tmp] [(ndarray/* (ndarray/ones shape-vec) 8)] 1e-5)
    (let [data-deriv (ndarray/* (ndarray/* (ndarray/ones shape-vec) 4) exp-tmp)
          exp-deriv (ndarray/* (ndarray/* (ndarray/ones shape-vec) 8)
                               (ndarray/* (ndarray/ones shape-vec) (Math/log 2)))]
      (check-symbolic-backward test
                               [data-tmp exp-tmp]
                               [(ndarray/ones shape-vec)]
                               [data-deriv exp-deriv] 1e-5))))

(deftest test-pow-fn
  (let [shape-vec [3 4]
        exp (sym/variable "exp")
        y (sym/** exp 2)
        x (ndarray/* (ndarray/ones shape-vec) 3)]
    (check-symbolic-forward y [x] [(ndarray/* (ndarray/ones shape-vec) 9)] 1e-5)
    ;; deriv is 2x
    (check-symbolic-backward y
                             [x]
                             [(ndarray/ones shape-vec)]
                             [(-> (ndarray/ones shape-vec)
                                  (ndarray/* 6))]
                             1e-5)))

(defn check-scalar-operation
  [operator data-vec num expected]
  (let [data (sym/variable "datas")
        shape-vec [2 2]
        test (operator data num)
        exec (sym/simple-bind test (context/default-context) {"datas" shape-vec})
        _ (executor/set-arg exec "datas" data-vec)
        output (-> (executor/forward exec) (executor/outputs) first)]
    (is (approx= 1e-5 expected output))
    (is (= [0 0 0 0]) (-> (executor/backward exec (ndarray/ones shape-vec))
                          (executor/get-grad "datas")
                          (ndarray/->vec)))))

(defn check-symbol-operation
  [operator data-vec-1 data-vec-2 expected]
  (let [data (sym/variable "datas")
        data2 (sym/variable "datas2")
        shape-vec [2 2]
        test (operator data data2)
        exec (sym/simple-bind test (context/default-context) {"datas" shape-vec "datas2" shape-vec})
        _ (executor/set-arg exec "datas" data-vec-1)
        _ (executor/set-arg exec "datas2" data-vec-2)
        output (-> (executor/forward exec) (executor/outputs) first)]
    (is (approx= 1e-5 expected output))
    _ (executor/backward exec (ndarray/ones shape-vec))
    (is (= [0 0 0 0]) (-> (executor/get-grad exec "datas") (ndarray/->vec)))
    (is (= [0 0 0 0]) (-> (executor/get-grad exec "datas2") (ndarray/->vec)))))

(defn check-scalar-2-operation
  [operator data-vec expected]
  (let [data (sym/variable "datas")
        shape-vec [2 2]
        test (operator data 2)
        exec (sym/simple-bind test (context/default-context) {"datas" shape-vec})
        _ (executor/set-arg exec "datas" data-vec)
        output (-> (executor/forward exec) (executor/outputs) first)]
    (is (approx= 1e-5 expected output))
    (is (= [0 0 0 0]) (-> (executor/backward exec (ndarray/ones shape-vec))
                          (executor/get-grad "datas")
                          (ndarray/->vec)))))

(deftest test-scalar-equal
  (check-scalar-operation sym/equal [1 2 3 4] 2 [0 1 0 0]))

(deftest test-symbol-equal
  (check-symbol-operation sym/equal [1 2 3 4] [1 3 2 6] [1 0 0 0]))

(deftest test-scalar-equal-2
  (check-scalar-2-operation sym/equal [1 2 3 4] [0 1 0 0]))

(deftest test-scalar-not-equal
  (check-scalar-operation sym/not-equal [1 2 3 4] 2 [1 0 1 1]))

(deftest test-symbol-not-equal
  (check-symbol-operation sym/not-equal [1 2 3 4] [1 3 2 6] [0 1 1 1]))

(deftest test-scalar-not-equal-2
  (check-scalar-2-operation sym/not-equal [1 2 3 4] [1 0 1 1]))

(deftest test-scalar-greater
  (check-scalar-operation sym/> [1 2 3 4] 2 [0 0 1 1]))

(deftest test-symbol-greater
  (check-symbol-operation sym/> [1 2 3 4] [1 3 2 6] [0 0 1 0]))

(deftest test-scalar-greater-equal
  (check-scalar-operation sym/>= [1 2 3 4] 2 [0 1 1 1]))

(deftest test-symbol-greater-equal
  (check-symbol-operation sym/>= [1 2 3 4] [1 3 2 6] [1 0 1 0]))

(deftest test-scalar-lesser
  (check-scalar-operation sym/< [1 2 3 4] 2 [1 0 0 0]))

(deftest test-symbol-lesser
  (check-symbol-operation sym/< [1 2 3 4] [1 3 2 6] [0 1 0 1]))

(deftest test-scalar-lesser-equal
  (check-scalar-operation sym/<= [1 2 3 4] 2 [1 1 0 0]))

(deftest test-symbol-lesser-equal
  (check-symbol-operation sym/<= [1 2 3 4] [1 3 2 6] [1 1 0 1]))

(deftest test-embedding
  (let [data (sym/variable "data")
        embed (sym/embedding "embed" {:data data :input-dim 10 :output-dim 4})]
    (println "Embedded symbol:" (sym/to-json embed))))

(deftest test-binary-duplicate-input
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5)
        arr-data (ndarray/copy data-tmp)
        arr-grad (ndarray/* (ndarray/ones shape-vec) 3)
        out-grad (ndarray/ones shape-vec)
        square (sym/* data data)
        exec-square (sym/bind square (context/default-context) arr-data arr-grad)]
    (executor/forward exec-square)
    (approx= 1e-6 (ndarray/* data-tmp data-tmp) (-> (executor/outputs exec-square) (first)))
    (executor/backward exec-square out-grad)
    (approx= 1e-6 (ndarray/* data-tmp 2) arr-grad)))

(deftest test-sign
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5)
        arr-data (ndarray/copy data-tmp)
        arr-grad (ndarray/* (ndarray/ones shape-vec) 3)

        test (sym/sign data)
        exec-test (sym/bind test (context/default-context) [arr-data] [arr-grad])]
    (is (test-util/approx= 1e-6
                           (-> (ndarray/sign data-tmp) (ndarray/->vec))
                           (-> exec-test (executor/forward) (executor/outputs) first (ndarray/->vec))))
    (executor/backward exec-test (ndarray/* (ndarray/ones shape-vec) 2))
    (is (approx= 1e-6 (ndarray/zeros shape-vec) arr-grad))))

(deftest test-round-ceil-floor
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5.543)
        arr-data (ndarray/copy data-tmp)
        arr-grad (ndarray/* (ndarray/ones shape-vec) 2)

        test (-> (sym/round data)
                 (sym/+ (sym/ceil data))
                 (sym/+ (sym/floor data)))
        exec-test (sym/bind test (context/default-context) [arr-data])]
    (is (approx= 1e-6
                 (-> (ndarray/round data-tmp)
                     (ndarray/+ (ndarray/ceil data-tmp))
                     (ndarray/+ (ndarray/floor data-tmp)))
                 (-> (executor/forward exec-test) (executor/outputs) (first))))))

(deftest test-rsqrt-cos-sin
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5)
        arr-data (ndarray/copy data-tmp)
        arr-grad (ndarray/* (ndarray/ones shape-vec) 3)

        test (-> (sym/rsqrt data)
                 (sym/+ (sym/cos data))
                 (sym/+ (sym/sin data)))
        exec-test (sym/bind test (context/default-context) [arr-data])]
    (is (approx= 1e-6
                 (-> (ndarray/rsqrt data-tmp)
                     (ndarray/+ (ndarray/cos data-tmp))
                     (ndarray/+ (ndarray/sin data-tmp)))
                 (-> (executor/forward exec-test) (executor/outputs) (first))))))

(deftest test-maximum
  (let [data1 (sym/variable "data")
        data2 (sym/variable "data")
        shape-vec [3 4]
        data-tmp1 (random/uniform 0 100 shape-vec)
        data-tmp2 (random/uniform 0 100 shape-vec)

        arr-data1 (ndarray/copy data-tmp1)
        arr-data2 (ndarray/copy data-tmp2)

        test (sym/max data1 data2)
        exec-test (sym/bind test (context/default-context) [arr-data1 arr-data2])
        out (-> (executor/forward exec-test) (executor/outputs) (first))]
    (is (approx= 1e-6
                 (mapv max (ndarray/->vec data-tmp1) (ndarray/->vec data-tmp2))
                 out))))

(deftest test-minimun
  (let [data1 (sym/variable "data")
        data2 (sym/variable "data")
        shape-vec [3 4]
        data-tmp1 (random/uniform 0 100 shape-vec)
        data-tmp2 (random/uniform 0 100 shape-vec)

        arr-data1 (ndarray/copy data-tmp1)
        arr-data2 (ndarray/copy data-tmp2)

        test (sym/min data1 data2)
        exec-test (sym/bind test (context/default-context) [arr-data1 arr-data2])
        out (-> (executor/forward exec-test) (executor/outputs) (first))]
    (is (approx= 1e-6
                 (mapv min (ndarray/->vec data-tmp1) (ndarray/->vec data-tmp2))
                 out))))

(deftest test-transpose
  (let [data (sym/variable "data")
        test (sym/transpose data)
        shape-vec [3 4]
        ctx (context/default-context)
        arr-data (random/uniform 0 100 shape-vec ctx)
        trans (ndarray/transpose (ndarray/copy arr-data))
        exec-test (sym/bind test ctx {"data" arr-data})
        out     (->  (executor/forward exec-test)
                     (executor/outputs)
                     (first))]
    (is (approx= 1e-6 trans out))
    (is (= [4 3] (mx-shape/->vec (ndarray/shape out))))))

(deftest test-smooth-l1-and-make-loss
  (let [data (sym/variable "data")
        smooth-l1 (sym/smooth-l1 {:data data :scalar 1.0})
        loss (sym/make-loss {:data smooth-l1})
        shape-vec [2 6]
        ctx (context/default-context)
        input (ndarray/array [-3.5 -2.5 -1.5 -0.5 -0.3 -0.1
                              0.1 0.3 0.5 1.5 2.5 3.5] shape-vec)
        grad (ndarray/empty shape-vec)
        arr-tmp [3.0 2.0 1.0 0.125 0.045 0.005
                 0.005 0.045 0.125 1.0 2.0 3.0]
        grad-tmp [-1.0 -1.0 -1.0 -0.5 -0.3 -0.1
                  0.1 0.3 0.5 1.0 1.0 1.0]
        exec-test (sym/bind loss ctx {:data input} {:data grad})
        out (-> (executor/forward exec-test) (executor/outputs) first)]
    (is (approx= 1e-6 arr-tmp out))
    (executor/backward exec-test)
    (is (approx= 1e-6 grad-tmp grad))))

(deftest test-maximum-minimum-scalar
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 2)
        arr-data (ndarray/copy data-tmp)
        test (-> (sym/max data 3)
                 (sym/+ (sym/max data 9))
                 (sym/+ (sym/min data 5))
                 (sym/+ (sym/min data 4)))
        exec-test (sym/bind test (context/default-context) [arr-data])]
    ;; 3 + 9 + 2 + 2
    (is (approx= 1e-6 (ndarray/* (ndarray/ones shape-vec) 16) (-> (executor/forward exec-test) (executor/outputs) (first))))))

(deftest test-abs
  (let [data (sym/variable "data")
        shape-vec [3 4]
        data-tmp (ndarray/* (ndarray/ones shape-vec) 5)
        arr-data (ndarray/copy data-tmp)
        arr-grad (ndarray/* (ndarray/ones shape-vec) 3)
        test (sym/abs data)
        exec-test (sym/bind test (context/default-context) arr-data arr-grad)]
    (is (approx= 1e-6 (ndarray/abs data-tmp) (-> (executor/forward exec-test) (executor/outputs) (first))))

    (let [out-grad (ndarray/* (ndarray/ones shape-vec) 2)
          npout-grad (ndarray/* out-grad (ndarray/sign data-tmp))]
      (executor/backward exec-test out-grad)
      (is (approx= 1e-6 npout-grad arr-grad)))));; configure A: input --> conv --> deconv --> output.
  ;;  the convolution and deconvoluiton has similar parameter which ensure
  ;;  the input shape is the same as output, and the same weights between conv
  ;;  and deconv;
  ;;  If the input value of forward() and backwrad() is the same, then
;;  the output value of them should also the same;

(defn check-deconvolution-forward-backward [{:keys [input-shape-vec num-filter kernel stride pad]}]
  (let [data (sym/variable "data")
        conv (sym/convolution "conv" {:data data :kernel kernel :stride stride
                                      :pad pad :num-filter num-filter :no-bias "true"})
        deconv (sym/deconvolution "deconv" {:data conv :kernel kernel :stride stride
                                            :pad pad :num-filter num-filter :no-bias "true"})
        arg-names (sym/list-arguments deconv)
        arg-shape-vecs (first (sym/infer-shape deconv {:data input-shape-vec}))
        input-data (random/uniform -5 5 input-shape-vec)
        out-grad input-data
        conv-weight (random/normal 0 1 [num-filter (second input-shape-vec) (first kernel) (last kernel)])
        args {:data input-data :conv-weight conv-weight :deconv-weight conv-weight}
        args-grad (mapv #(ndarray/empty %) arg-shape-vecs)
        exec (sym/bind deconv (context/default-context) args args-grad)
        out (-> (executor/forward exec) (executor/outputs) first)]
    (executor/backward exec out-grad)
    (is (approx= 1e-3 (ndarray/->vec out) (ndarray/->vec (first args-grad))))))

(deftest test-deconvolution-forward-and-backward
  (check-deconvolution-forward-backward {:input-shape-vec [1 1 5 5] :num-filter 1 :kernel [3 3] :stride [1 1] :pad [1 1]})
  (check-deconvolution-forward-backward {:input-shape-vec [32 3 28 28] :num-filter 3 :kernel [3 3] :stride [1 1] :pad [1 1]})
  ;; commented out to make the tests fast
  #_(check-deconvolution-forward-backward {:input-shape-vec [10 3 403 403] :num-filter 3 :kernel [7 7] :stride [5 5] :pad [2 2]}))

;;  configure A: input --> conv --> output.
;;  configure B: input --> deconv --> output
;;  the convolution and deconvoluiton has similar parameter which ensure
;;   the input shape is the same as output;
;;   During backward(), if the input of A equals output of B, and the output
;;    of A equals input of B, then the grad of weight should be the same;

(defn check-deconvolution-gradient [{:keys [input-shape-vec num-filter pad]}]
  (let [stride [1 1]
        kernel [(inc (* 2 (first pad))) (inc (* 2 (second pad)))]
        data-conv (sym/variable "data_conv")
        conv (sym/convolution "conv" {:data data-conv :kernel kernel :stride stride
                                      :pad pad :num-filter num-filter :no-bias "true"})
        data-deconv (sym/variable "data_deconv")
        deconv (sym/deconvolution "deconv" {:data data-deconv :kernel kernel :stride stride
                                            :pad pad :num-filter num-filter :no-bias true})
        conv-data (random/uniform -5 5 input-shape-vec)
        conv-args {"data_conv" conv-data "conv_weight" (random/normal 0 1 [num-filter (second input-shape-vec) (first kernel) (second kernel)])}
        conv-args-grad [(ndarray/zeros (-> conv-data (ndarray/shape) (ndarray/->vec)))
                        (ndarray/zeros [num-filter (second input-shape-vec) (first kernel) (second kernel)])]
        exec-conv (sym/bind conv (context/default-context) conv-args conv-args-grad)
        conv-out-grad (random/normal 0 2 (-> (executor/outputs exec-conv) (first) (ndarray/shape) (mx-shape/->vec)))]
    (executor/forward exec-conv)
    (executor/backward exec-conv conv-out-grad)

    (let [deconv-data conv-out-grad
          deconv-args {"data_deconv" deconv-data "deconv_weight" (get conv-args "conv_weight")}
          deconv-args-grad [(ndarray/zeros (-> deconv-data (ndarray/shape) (mx-shape/->vec)))
                            (ndarray/zeros [num-filter (second input-shape-vec) (first kernel) (second kernel)])]
          exec-deconv (sym/bind deconv (context/default-context) deconv-args deconv-args-grad)
          deconv-out-grad conv-data]
      (executor/forward exec-deconv)
      (executor/backward exec-deconv deconv-out-grad)

      (is (approx= 1e-4 (ndarray/->vec (second conv-args-grad)) (ndarray/->vec (second deconv-args-grad)))))))

(deftest test-deconvolution-gradient
  (check-deconvolution-gradient {:input-shape-vec [1 3 5 5] :num-filter 3 :pad [1 1]}))

(defn check-nearest-up-sampling-with-shape [{:keys [shape-vecs scale root-scale]}]
  (let [arr (zipmap (map #(str "arg_" %) (range 0 (count shape-vecs)))
                    (map #(random/uniform -10 10 %) shape-vecs))
        arr-grad (zipmap (map #(str "arg_" %) (range 0 (count shape-vecs)))
                         (map #(ndarray/zeros %) shape-vecs))
        up-args (mapv #(sym/variable (str "arg_" %)) (range 0 (count shape-vecs)))
        up (sym/up-sampling "up-sampling" nil up-args {:sample-type "nearest" :scale root-scale})
        exec (sym/bind up (context/default-context) arr arr-grad)]
    (executor/forward exec)
    (executor/backward exec (executor/outputs exec))
    (doseq [k (range 0 (count shape-vecs))]
      (let [k-name (str "arg_" k)
            expected (->> (get arr k-name) (ndarray/->vec) (mapv #(* % (Math/pow root-scale 2) (Math/pow scale (*  2 k)))))
            real (-> (get arr-grad k-name) (ndarray/->vec))]
        (is (approx= 0.1 expected real))))))

(deftest test-nearest-upsampling
  (doall (for [root-scale (range 1 4)
               scale (range 1 4)
               num-shape (range 1 4)
               base (range 1 4)]
           (let [shape-vecs (mapv (fn [i] [1 3 (* base root-scale (int (Math/pow scale (- (dec num-shape) i))))
                                           (* base root-scale (int (Math/pow scale (- (dec num-shape) i))))])
                                  (range 0 num-shape))]
             (check-nearest-up-sampling-with-shape {:shape-vecs shape-vecs :scale scale :root-scale root-scale})))))
