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
            [org.apache.clojure-mxnet.primitives :as primitives]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.test-util :as test-util]
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

(deftest test->int-option
  (let [x (util/->int-option 4.5)]
    (is (instance? Option x))
    (is (= 4 (.get x)))))

(deftest test-empty->int-option
  (let [x (util/->int-option nil)]
    (is (instance? Option x))
    (is (.isEmpty x))))

(deftest test-option->value
  (is (= 2 (-> (util/->option 2)
               (util/option->value)))))

(deftest test-keyword->snake-case
  (is (= ["foo_bar" "foo2" "bar_bar"]
         (mapv util/keyword->snake-case [:foo_bar :foo2 :bar-bar]))))

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

  (is (primitives/primitive? (util/coerce-param 1.0 #{"org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE"})))
  (is (primitives/primitive? (util/coerce-param (float 1.0) #{"org.apache.mxnet.MX_PRIMITIVES$MX_PRIMITIVE_TYPE"})))

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

(deftest test-to-array-nd
  (let [a1 (util/to-array-nd '(1))
        a2 (util/to-array-nd [1.0 2.0])
        a3 (util/to-array-nd [[3.0] [4.0]])
        a4 (util/to-array-nd [[[5 -5]]])]
    (is (= 1 (alength a1)))
    (is (= [1] (->> a1 vec)))
    (is (= 2 (alength a2)))
    (is (= 2.0 (aget a2 1)))
    (is (= [1.0 2.0] (->> a2 vec)))
    (is (= 2 (alength a3)))
    (is (= 1 (alength (aget a3 0))))
    (is (= 4.0 (aget a3 1 0)))
    (is (= [[3.0] [4.0]] (->> a3 vec (mapv vec))))
    (is (= 1 (alength a4)))
    (is (= 1 (alength (aget a4 0))))
    (is (= 2 (alength (aget a4 0 0))))
    (is (= 5 (aget a4 0 0 0)))
    (is (= [[[5 -5]]] (->> a4 vec (mapv vec) (mapv #(mapv vec %)))))))

(deftest test-nd-seq-shape
  (is (= [1] (util/nd-seq-shape '(5))))
  (is (= [2] (util/nd-seq-shape [1.0 2.0])))
  (is (= [3] (util/nd-seq-shape [1 1 1])))
  (is (= [2 1] (util/nd-seq-shape [[3.0] [4.0]])))
  (is (= [1 3 2] (util/nd-seq-shape [[[5 -5] [5 -5] [5 -5]]]))))

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

  (is (instance? Double (util/coerce-return (primitives/mx-double 3))))
  (is (= 3.0 (util/coerce-return (primitives/mx-double 3))))
  (is (instance? Float (util/coerce-return (primitives/mx-float 2))))
  (is (= 2.0 (util/coerce-return (primitives/mx-float 2))))

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

(deftest test-approx=
  (let [data1 [1 1 1 1]
        data2 [1 1 1 1 9 9 9 9]
        data3 [1 1 1 2]]
    (is (not (test-util/approx= 1e-9 data1 data2)))
    (is (test-util/approx= 2 data1 data3))))

(deftest test-map->scala-tuple-seq
  ;; convert as much, and pass-through the rest
  (is (nil? (util/map->scala-tuple-seq nil)))
  (is (= "List()"
         (str (util/map->scala-tuple-seq {}))
         (str (util/map->scala-tuple-seq []))
         (str (util/map->scala-tuple-seq '()))))
  (is (= "List(a, b)" (str (util/map->scala-tuple-seq ["a" "b"]))))
  (is (= "List((a,b), (c,d), (e,f), (a_b,g), (c_d,h), (e_f,i))"
         (str (util/map->scala-tuple-seq {:a "b", 'c "d", "e" "f"
                                          :a-b "g", 'c-d "h", "e-f" "i"}))))
  (let [nda (util/map->scala-tuple-seq {:a-b (ndarray/ones [1 2])})]
    (is (= "a_b" (._1 (.head nda))))
    (is (= [1.0 1.0] (ndarray/->vec (._2 (.head nda)))))))

(deftest test-forms->scala-fn
  (let [scala-fn (util/forms->scala-fn
                  (def x 1)
                  (def y 2)
                  {:x x :y y})]
    (is (= {:x 1 :y 2} (.apply scala-fn)))))
