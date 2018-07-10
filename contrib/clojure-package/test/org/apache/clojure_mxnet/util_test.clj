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

(ns org.apache.clojure-mxnet.util-test
  (:require [clojure.test :refer :all]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet Shape NDArrayFuncReturn NDArray)
           (scala.collection Map Set)
           (scala.collection.mutable ArrayBuffer)
           (scala.collection.immutable List IndexedSeq ListMap Vector)
           (scala Option Tuple1 Tuple2 Tuple3)))

(deftest test-empty-list
  (let [x (util/empty-list)]
    (is (instance? List x))
    (is (true? (.isEmpty x)))))

(deftest test-empty-map
  (let [x (util/empty-map)]
    (is (instance? Map x))
    (is (true? (.isEmpty x)))))

(deftest test-indexed-seq
  (let [x (util/empty-indexed-seq)]
    (is (instance? IndexedSeq x))
    (is (true? (.isEmpty x)))))

(deftest test-empty-list-map
  (let [x (util/empty-list-map)]
    (is (instance? ListMap x))
    (is (true? (.isEmpty x)))))

(deftest test->option
  (let [x (util/->option 1)]
    (is (instance? Option x))
    (is (= 1 (.get x)))))

(deftest test-option->value
  (is (= 2 (-> (util/->option 2)
               (util/option->value)))))

(deftest test-keyword->snake-case
  (is (= [:foo-bar :foo2 :bar-bar])
      (util/keyword->snake-case [:foo_bar :foo2 :bar-bar])))

(deftest test-convert-tuple
  (is (instance? Tuple1 (util/convert-tuple [1])))
  (is (instance? Tuple2 (util/convert-tuple [1 2])))
  (is (instance? Tuple3 (util/convert-tuple [1 2 3]))))

(deftest test-convert-by-shape
  (let [x (util/convert-by-shape {:a [100] :b "hi"})]
    (is (instance? Shape (:a x)))
    (is (= "hi" (:b x)))))

(deftest tuple-convert-by-param-name
  (let [x (util/tuple-convert-by-param-name {:foo [100] :kernel [3 3] :bar "hi"})]
    (is (= "(3,3)" (:kernel x)))
    (is (= [100] (:foo x)))
    (is (= "hi" (:bar x)))))

(deftest test-io-convert-by-param-name
  (let [x (util/io-convert-by-param-name {:input-shape [10 10] :freeze? true :foo 1})]
    (is (= "(10,10)" (:input-shape x)))
    (is (= "True" (:freeze? x)))
    (is (= "1" (:foo x)))))

(deftest test-convert-map
  (let [x (util/convert-map {:a [10] :b 1 :foo-bar 2})]
    (is (instance? Map x))
    (is (= "Set(a, b, foo_bar)" (-> x (.keys) str)))))

(deftest test-convert-vector
  (let [x (util/convert-vector [1 2 3])]
    (is (instance? List x))
    (is (= "List(1, 2, 3)" (str x)))))

(deftest test-vec->set
  (let [x (util/vec->set [1 2 3])]
    (is (instance? Set x))
    (is (= "Set(1, 2, 3)" (str x)))))

(deftest test-vec->indexed-seq
  (let [x (util/vec->indexed-seq [1 2 3])]
    (is (instance? Vector x))
    (is (= "Vector(1, 2, 3)" (str x)))))

(deftest test-scala-function
  (let [s-fn (util/scala-fn (fn [x] (+ x 2)))]
    (is (= 4 (util/apply-scala-fn s-fn 2)))))

(deftest test-coerce-param
  (is (instance? Map (util/coerce-param {:x 1} #{"scala.collection.immutable.Map"})))
  (is (map? (util/coerce-param {:x 1} #{"float"})))

  (is (float? (util/coerce-param 1 #{"float"})))

  (is (instance? List (util/coerce-param (ndarray/ones [3]) #{"scala.collection.Seq"})))
  (is (instance? List (util/coerce-param (sym/variable "a") #{"scala.collection.Seq"})))
  (is (instance? List (util/coerce-param [1 2] #{"scala.collection.Seq"})))
  (is (instance? List (util/coerce-param [] #{"scala.collection.Seq"})))

  (is (= "[I"  (->> (util/coerce-param [1 2] #{"int<>"}) str (take 2) (apply str))))
  (is (= "[F"  (->> (util/coerce-param [1 2] #{"float<>"}) str (take 2) (apply str))))
  (is (= "[L"  (->> (util/coerce-param [1 2] #{"java.lang.String<>"}) str (take 2) (apply str))))

  (is (= 1 (util/coerce-param 1 #{"unknown"}))))

(deftest test-nil-or-coerce-param
  (is (instance? Map (util/nil-or-coerce-param {:x 1} #{"scala.collection.immutable.Map"})))
  (is (nil? (util/coerce-param nil #{"scala.collection.immutable.Map"}))))

(deftest test-scala-map->map
  (is (= {"a" 1} (-> (util/convert-map {:a 1})
                     (util/scala-map->map)))))

(deftest test-buffer->vec
  (is (= [] (util/buffer->vec (ArrayBuffer.)))))

(deftest test-scala-vector->vec
  (is (= [1 2 3] (util/scala-vector->vec
                  (util/vec->indexed-seq [1 2 3])))))

(deftest test-scala-iterator->seq
  (is (= [1 2 3] (-> (util/vec->indexed-seq [1 2 3])
                     (.iterator)
                     (util/scala-iterator->seq)))))

(deftest test-tuple->vec
  (is (= [1 2] (-> (util/convert-tuple [1 2])
                   (util/tuple->vec)))))

(deftest test-coerce-return
  (is (= [] (util/coerce-return (ArrayBuffer.))))
  (is (= [1 2 3] (util/coerce-return (util/vec->indexed-seq [1 2 3]))))
  (is (instance? NDArray
                 (util/coerce-return
                  (new NDArrayFuncReturn (into-array [(ndarray/zeros [3])])))))
  (is (= {"x" 1} (util/coerce-return
                  (util/convert-map {:x 1}))))
  (is (= [1 2] (util/coerce-return
                (util/convert-tuple [1 2]))))
  (is (= [1 2 3] (util/coerce-return
                  (util/convert-tuple [1 2 3]))))
  (is (= "foo" (util/coerce-return "foo"))))

(deftest test-translate-keyword-shape
  (let [[name shape]  (util/translate-keyword-shape [:foo-a [5]])]
    (is (= name "foo_a"))
    (is (instance? Shape shape))
    (is (= "(5)" (str shape)))))

(deftest test-map->tuple
  (let [x (util/map->tuple {:foo-a [5]})]
    (is (instance? Tuple2 (first x)))
    (is (= "(foo_a,(5))" (str (first x))))))

(deftest test-list-map
  (let [x (util/list-map {:x 1 :y 2})]
    (is (instance? ListMap x))
    (is (= "Map(x -> 1, y -> 2)" (str x)))))

(s/def ::x string?)

(deftest test-validate
  (is (nil? (util/validate! string? "foo" "Not a string!")))
  (is (thrown-with-msg? Exception #"Not a string!" (util/validate! ::x 1 "Not a string!"))))
