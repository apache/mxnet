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

(ns org.apache.clojure-mxnet.resource-scope-test
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [clojure.test :refer :all]))


(deftest test-resource-scope-with-ndarray
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (ndarray/ones [3 1])
                          temp-y (ndarray/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                      (swap! native-resources assoc :temp-y temp-y)
                      (ndarray/+ temp-x 1)))]
    (is (true? (.isDisposed (:temp-x @native-resources))))
    (is (true? (.isDisposed (:temp-y @native-resources))))
    (is (false? (.isDisposed return-val)))
    (is (false? (.isDisposed x)))
    (is (= [2.0 2.0 2.0] (ndarray/->vec return-val)))))

(deftest test-nested-resource-scope-with-ndarray
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (ndarray/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                     (resource-scope/using
                      (let [temp-y (ndarray/ones [3 1])]
                        (swap! native-resources assoc :temp-y temp-y)))))]
    (is (true? (.isDisposed (:temp-y @native-resources))))
    (is (true? (.isDisposed (:temp-x @native-resources))))
    (is (false? (.isDisposed x)))))

(deftest test-resource-scope-with-sym
  (let [native-resources (atom {})
        x (sym/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (sym/ones [3 1])
                          temp-y (sym/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                      (swap! native-resources assoc :temp-y temp-y)
                      (sym/+ temp-x 1)))]
    (is (true? (.isDisposed (:temp-x @native-resources))))
    (is (true? (.isDisposed (:temp-y @native-resources))))
    (is (false? (.isDisposed return-val)))
    (is (false? (.isDisposed x)))))
