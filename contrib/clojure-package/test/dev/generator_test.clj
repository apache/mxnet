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

(ns dev.generator-test
  (:require [clojure.test :refer :all]
            [dev.generator :as gen]))

(deftest test-clojure-case
  (is (= "foo-bar" (gen/clojure-case "FooBar")))
  (is (= "foo-bar-baz" (gen/clojure-case "FooBarBaz")))
  (is (= "foo-bar-baz" (gen/clojure-case "FOOBarBaz")))
  (is (= "foo-bar" (gen/clojure-case "foo_bar")))
  (is (= "foo-bar" (gen/clojure-case "Foo_Bar")))
  (is (= "div+" (gen/clojure-case "/+"))))

(defn ndarray-reflect-info [name]
  (->> gen/ndarray-public-no-default
       (filter #(= name (str (:name %))))
       first))

(defn symbol-reflect-info [name]
  (->> gen/symbol-public-no-default
       (filter #(= name (str (:name %))))
       first))

(deftest test-symbol-transform-param-name
  (let [params ["java.lang.String"
                "scala.collection.immutable.Map"
                "scala.collection.Seq"
                "scala.collection.immutable.Map"]
        transformed-params ["sym-name"
                            "kwargs-map"
                            "symbol-list"
                            "kwargs-map"]]
    (is (= transformed-params (gen/symbol-transform-param-name params)))
    (is (= transformed-params (gen/symbol-transform-param-name
                               (:parameter-types (symbol-reflect-info "floor")))))))

(deftest test-ndarray-transform-param-name
  (let [params ["scala.collection.immutable.Map"
                "scala.collection.Seq"]
        transformed-params ["kwargs-map" "& nd-array-and-params"]]
    (is (= transformed-params (gen/ndarray-transform-param-name params)))
    (is (= transformed-params (gen/ndarray-transform-param-name
                               (:parameter-types (ndarray-reflect-info "sqrt")))))))

(deftest test-has-variadic?
  (is (false? (gen/has-variadic? ["sym-name" "kwargs-map" "symbol-list" "kwargs-map-1"])))
  (is (true? (gen/has-variadic? ["kwargs-map" "& nd-array-and-params"]))))

(deftest test-increment-param-name
  (is (= "foo-1" (gen/increment-param-name "foo")))
  (is (= "foo-2" (gen/increment-param-name "foo-1"))))

(deftest test-rename-duplicate-params
  (is (= ["foo" "bar" "baz"] (gen/rename-duplicate-params ["foo" "bar" "baz"])))
  (is (= ["foo" "bar" "bar-1"] (gen/rename-duplicate-params ["foo" "bar" "bar"]))))

(deftest test-is-symbol-hand-gen?
  (is (not (false? (gen/is-symbol-hand-gen? (symbol-reflect-info "max")))))
  (is (not (false? (gen/is-symbol-hand-gen? (symbol-reflect-info "Variable")))))
  (is (false? (gen/is-symbol-hand-gen? (symbol-reflect-info "sqrt")))))

(deftest test-is-ndarray-hand-gen?
  (is (not (false? (gen/is-ndarray-hand-gen? (ndarray-reflect-info "zeros")))))
  (is (false? (gen/is-ndarray-hand-gen? (ndarray-reflect-info "sqrt")))))

(deftest test-public-by-name-and-param-count
  (let [lrn-info (get (gen/public-by-name-and-param-count gen/symbol-public-to-gen)
                      (symbol "LRN"))]
    (is (= 4 (-> lrn-info keys first)))
    (is (= "LRN" (-> lrn-info vals ffirst :name str)))))

(deftest test-symbol-vector-args
  (is (= `(if (clojure.core/map? kwargs-map-or-vec-or-sym)
            (util/empty-list)
            (util/coerce-param
             kwargs-map-or-vec-or-sym
             #{"scala.collection.Seq"}))) (gen/symbol-vector-args)))

(deftest test-symbol-map-args
  (is (= `(if (clojure.core/map? kwargs-map-or-vec-or-sym)
            (org.apache.clojure-mxnet.util/convert-symbol-map
             kwargs-map-or-vec-or-sym)
            nil))
      (gen/symbol-map-args)))

(deftest test-add-symbol-arities
  (let [params (map symbol ["sym-name" "kwargs-map" "symbol-list" "kwargs-map-1"])
        function-name (symbol "foo")
        [ar1 ar2 ar3] (gen/add-symbol-arities params function-name)]
    (is (= '([sym-name attr-map kwargs-map]
             (foo
              sym-name
              (util/convert-symbol-map attr-map)
              (util/empty-list)
              (util/convert-symbol-map kwargs-map)))
           ar1))
    (is (= '([sym-name kwargs-map-or-vec-or-sym]
             (foo
              sym-name
              nil
              (if
               (clojure.core/map? kwargs-map-or-vec-or-sym)
                (util/empty-list)
                (util/coerce-param
                 kwargs-map-or-vec-or-sym
                 #{"scala.collection.Seq"}))
              (if
               (clojure.core/map? kwargs-map-or-vec-or-sym)
                (org.apache.clojure-mxnet.util/convert-symbol-map
                 kwargs-map-or-vec-or-sym)
                nil))))
        ar2)
    (is (= '([kwargs-map-or-vec-or-sym]
             (foo
              nil
              nil
              (if
               (clojure.core/map? kwargs-map-or-vec-or-sym)
                (util/empty-list)
                (util/coerce-param
                 kwargs-map-or-vec-or-sym
                 #{"scala.collection.Seq"}))
              (if
               (clojure.core/map? kwargs-map-or-vec-or-sym)
                (org.apache.clojure-mxnet.util/convert-symbol-map
                 kwargs-map-or-vec-or-sym)
                nil))))
        ar3)))

(deftest test-gen-symbol-function-arity
  (let [op-name (symbol "$div")
        op-values {1 [{:name (symbol "$div")
                       :return-type "org.apache.mxnet.Symbol,"
                       :declaring-class "org.apache.mxnet.Symbol,"
                       :parameter-types ["org.apache.mxnet.Symbol"],
                       :exception-types [],
                       :flags #{:public}}
                      {:name (symbol "$div") :return-type "org.apache.mxnet.Symbol,"
                       :declaring-class "org.apache.mxnet.Symbol,"
                       :parameter-types ["java.lang.Object"],
                       :exception-types [],
                       :flags #{:public}}]}
        function-name (symbol "div")]
    (is (= '(([sym sym-or-Object]
              (util/coerce-return
               (.$div
                sym
                (util/nil-or-coerce-param
                 sym-or-Object
                 #{"org.apache.mxnet.Symbol" "java.lang.Object"}))))))
        (gen/gen-symbol-function-arity op-name op-values function-name))))

(deftest test-gen-ndarray-function-arity
  (let [op-name (symbol "$div")
        op-values {1 [{:name (symbol "$div")
                       :return-type "org.apache.mxnet.NDArray,"
                       :declaring-class "org.apache.mxnet.NDArray,"
                       :parameter-types ["float"],
                       :exception-types [],
                       :flags #{:public}}
                      {:name (symbol "$div")
                       :return-type "org.apache.mxnet.NDArray,"
                       :declaring-class "org.apache.mxnet.NDArray,"
                       :parameter-types ["org.apache.mxnet.NDArray"],
                       :exception-types [],
                       :flags #{:public}}]}]
    (is (= '(([ndarray num-or-ndarray]
              (util/coerce-return
               (.$div
                ndarray
                (util/coerce-param
                 num-or-ndarray
                 #{"float" "org.apache.mxnet.NDArray"}))))))
        (gen/gen-ndarray-function-arity op-name op-values))))

(deftest test-write-to-file
  (testing "symbol"
    (let [fname "test/test-symbol.clj"
          _ (gen/write-to-file [(first gen/all-symbol-functions)]
                               gen/symbol-gen-ns
                               fname)
          good-contents (slurp "test/good-test-symbol.clj")
          contents (slurp fname)]
      (is (= good-contents contents))))

  (testing "ndarray"
    (let [fname "test/test-ndarray.clj"
          _ (gen/write-to-file [(first gen/all-ndarray-functions)]
                               gen/ndarray-gen-ns
                               fname)
          good-contents (slurp "test/good-test-ndarray.clj")
          contents (slurp fname)]
      (is (= good-contents contents)))))
