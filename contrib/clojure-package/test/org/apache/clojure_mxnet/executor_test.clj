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

(ns org.apache.clojure-mxnet.executor-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.test-util :as test-util]
            [clojure.test :refer :all]))

(deftest test-bind
  (let [shape [100 30]
        lhs (sym/variable "lhs")
        rhs (sym/variable "rhs")
        ret (sym/+ lhs rhs)]
    (is (= ["lhs" "rhs"] (sym/list-arguments ret)))

    (let [lhs-arr (random/uniform -10 10 shape)
          rhs-arr (random/uniform -10 10 shape)
          lhs-grad (ndarray/empty shape)
          rhs-grad (ndarray/empty shape)
          exec (sym/bind ret (context/default-context) [lhs-arr rhs-arr] [lhs-grad rhs-grad])
          exec2 (sym/bind ret (context/default-context) [lhs-arr rhs-arr])
          exec3 (sym/bind ret (context/default-context) {"rhs" rhs-arr "lhs" lhs-arr} {"lhs" lhs-grad "rhs" rhs-grad})]
      (executor/forward exec)
      (executor/forward exec2)
      (executor/forward exec3)
      (is (test-util/approx= 1e-6 (-> (ndarray/+ lhs-arr rhs-arr) ndarray/->vec) (-> (executor/outputs exec) first ndarray/->vec)))
      (is (test-util/approx= 1e-6 (-> (ndarray/+ lhs-arr rhs-arr) ndarray/->vec) (-> (executor/outputs exec2) first ndarray/->vec)))
      (is (test-util/approx= 1e-6 (-> (ndarray/+ lhs-arr rhs-arr) ndarray/->vec) (-> (executor/outputs exec3) first ndarray/->vec)))

      ;; test gradient
      (let [out-grad (ndarray/ones shape)
            lhs-grad2 out-grad
            rhs-grad2 out-grad]
        (executor/backward exec out-grad)
        (is (test-util/approx= 1e-6 (ndarray/->vec lhs-grad) (ndarray/->vec lhs-grad2)))
        (is (test-util/approx= 1e-6 (ndarray/->vec rhs-grad) (ndarray/->vec rhs-grad2)))))))

(deftest test-reshape
  (let [x (sym/variable "x")
        y (sym/fully-connected {:data x :num-hidden 4})
        exec (sym/simple-bind y (context/default-context) {"x" [5 4]})
        _ (executor/set-arg-arrays exec [1 1 0])
        new-exec (executor/reshape exec {"x" [3 4]})]
    (executor/forward new-exec)
    ;; test sub exec forward
    (is (every? #(= 4.0 %) (->> (executor/outputs new-exec)
                                (map ndarray/->vec)
                                first)))
    ;; test shared memory
    (is (= [4.0 4.0 4.0]) (->> (executor/outputs exec)
                               (map ndarray/->vec)
                               first
                               (take 3)))
    ;; test base exec forward
    (executor/forward exec)
    (is (every? #(= 4.0 %) (->> (executor/outputs exec)
                                (map ndarray/->vec)
                                first)))))
