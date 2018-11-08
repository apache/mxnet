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

(ns org.apache.clojure-mxnet.initializer-test
  (:require [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [clojure.test :refer :all]))

(defn exercise-initializer [init]
  (-> init
      (initializer/init-weight "test-weight" (ndarray/zeros [3 3])))

  (is (number?
       (-> init
           (initializer/apply "test-weight" (ndarray/zeros [3 3]))
           (ndarray/->vec)
           (first)))))

(deftest test-uniform
  (exercise-initializer (initializer/uniform))
  (exercise-initializer (initializer/uniform 0.8)))

(deftest test-normal
  (exercise-initializer (initializer/normal))
  (exercise-initializer (initializer/normal 0.2)))

(deftest test-xavier
  (exercise-initializer (initializer/xavier))
  (exercise-initializer (initializer/xavier {:rand-type "gaussian"
                                             :factor-type "in"
                                             :magnitude 2})))
