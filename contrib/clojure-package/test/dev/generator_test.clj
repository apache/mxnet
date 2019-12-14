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
            [dev.generator :as gen]
            [clojure.string :as string]))

(defn file-function-name [f]
  (->> (string/split (slurp f) #"\n")
       (take 33)
       last
       (string/trim)))

(deftest test-clojure-case
  (is (= "foo-bar" (gen/clojure-case "FooBar")))
  (is (= "foo-bar-baz" (gen/clojure-case "FooBarBaz")))
  (is (= "foo-bar-baz" (gen/clojure-case "FOOBarBaz")))
  (is (= "foo-bar" (gen/clojure-case "foo_bar")))
  (is (= "foo-bar" (gen/clojure-case "Foo_Bar")))
  (is (= "div+" (gen/clojure-case "/+"))))

(deftest fn-name->random-fn-name
  (is (= "poisson" (gen/fn-name->random-fn-name "-random-poisson")))
  (is (= "poisson-like" (gen/fn-name->random-fn-name "-sample-poisson"))))

(deftest remove-prefix
  (is (= "randint" (gen/remove-prefix "-random-" "-random-randint")))
  (is (= "exponential" (gen/remove-prefix "-sample-" "-sample-exponential"))))

(deftest in-namespace-random?
  (is (gen/in-namespace-random? "random_randint"))
  (is (gen/in-namespace-random? "sample_poisson"))
  (is (not (gen/in-namespace-random? "rnn")))
  (is (not (gen/in-namespace-random? "activation"))))

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

(deftest test-gen-op-info
  (testing "activation"
    (let [activation-info (gen/gen-op-info "Activation")]
      (is (= "activation" (:fn-name activation-info)))
      (is (string? (:fn-description activation-info)))
      (is (= 2 (-> activation-info :args count)))
      (is (= "" (:key-var-num-args activation-info)))

      (is (= "data" (-> activation-info :args first :name)))
      (is (= "NDArray-or-Symbol" (-> activation-info :args first :type)))
      (is (false? (-> activation-info :args first :optional?)))
      (is (nil? (-> activation-info :args first :default)))
      (is (string? (-> activation-info :args first :description)))

      (is (= "act-type" (-> activation-info :args second :name)))
      (is (= "'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'" (-> activation-info :args second :type)))
      (is (false? (-> activation-info :args second :optional?)))
      (is (nil? (-> activation-info :args second :default)))
      (is (string? (-> activation-info :args second :description)))))

  (testing "argmin"
    (let [argmin-info (gen/gen-op-info "argmin")]
      (is (= "argmin" (:fn-name argmin-info)))
      (is (= 3 (-> argmin-info :args count)))

      (is (= "data" (-> argmin-info :args (nth 0) :name)))
      (is (= "NDArray-or-Symbol" (-> argmin-info :args (nth 0) :type)))
      (is (false? (-> argmin-info :args (nth 0) :optional?)))

      (is (= "axis" (-> argmin-info :args (nth 1) :name)))
      (is (= "int or None" (-> argmin-info :args (nth 1) :type)))
      (is (= "'None'" (-> argmin-info :args (nth 1) :default)))
      (is (true? (-> argmin-info :args (nth 1) :optional?)))

      (is (= "keepdims" (-> argmin-info :args (nth 2) :name)))
      (is (= "boolean" (-> argmin-info :args (nth 2) :type)))
      (is (= "0" (-> argmin-info :args (nth 2) :default)))
      (is (true? (-> argmin-info :args (nth 2) :optional?)))))

  (testing "concat"
    (let [concat-info (gen/gen-op-info "Concat")]
      (is (= "concat" (:fn-name concat-info)))
      (is (= 3 (-> concat-info :args count)))
      (is (= "num-args" (:key-var-num-args concat-info)))

      (is (= "data" (-> concat-info :args (nth 0) :name)))
      (is (= "NDArray-or-Symbol[]" (-> concat-info :args (nth 0) :type)))
      (is (false? (-> concat-info :args (nth 0) :optional?)))

      (is (= "num-args" (-> concat-info :args (nth 1) :name)))
      (is (= "int" (-> concat-info :args (nth 1) :type)))
      (is (false? (-> concat-info :args (nth 1) :optional?)))

      (is (= "dim" (-> concat-info :args (nth 2) :name)))
      (is (= "int" (-> concat-info :args (nth 2) :type)))
      (is (= "'1'" (-> concat-info :args (nth 2) :default)))
      (is (true? (-> concat-info :args (nth 2) :optional?)))))

  (testing "convolution"
    (let [convolution-info (gen/gen-op-info "Convolution")]

      (is (= "convolution" (:fn-name convolution-info)))
      (is (= 14 (-> convolution-info :args count)))
      (is (= "" (:key-var-num-args convolution-info)))

      (is (= "data" (-> convolution-info :args (nth 0) :name)))
      (is (= "NDArray-or-Symbol" (-> convolution-info :args (nth 0) :type)))
      (is (false? (-> convolution-info :args (nth 0) :optional?)))

      (is (= "weight" (-> convolution-info :args (nth 1) :name)))
      (is (= "NDArray-or-Symbol" (-> convolution-info :args (nth 1) :type)))
      (is (false? (-> convolution-info :args (nth 1) :optional?)))

      (is (= "kernel" (-> convolution-info :args (nth 3) :name)))
      (is (= "Shape" (-> convolution-info :args (nth 3) :type)))
      (is (= "(tuple)" (-> convolution-info :args (nth 3) :spec)))
      (is (false? (-> convolution-info :args (nth 3) :optional?)))

      (is (= "stride" (-> convolution-info :args (nth 4) :name)))
      (is (= "Shape" (-> convolution-info :args (nth 4) :type)))
      (is (= "(tuple)" (-> convolution-info :args (nth 4) :spec)))
      (is (= "[]" (-> convolution-info :args (nth 4) :default)))
      (is (true? (-> convolution-info :args (nth 4) :optional?)))

      (is (= "num-filter" (-> convolution-info :args (nth 7) :name)))
      (is (= "int" (-> convolution-info :args (nth 7) :type)))
      (is (= "(non-negative)" (-> convolution-info :args (nth 7) :spec)))
      (is (false? (-> convolution-info :args (nth 7) :optional?)))

      (is (= "num-group" (-> convolution-info :args (nth 8) :name)))
      (is (= "int" (-> convolution-info :args (nth 8) :type)))
      (is (= "(non-negative)" (-> convolution-info :args (nth 8) :spec)))
      (is (= "1" (-> convolution-info :args (nth 8) :default)))
      (is (true? (-> convolution-info :args (nth 8) :optional?)))

      (is (= "workspace" (-> convolution-info :args (nth 9) :name)))
      (is (= "long" (-> convolution-info :args (nth 9) :type)))
      (is (= "(non-negative)" (-> convolution-info :args (nth 9) :spec)))
      (is (= "1024" (-> convolution-info :args (nth 9) :default)))
      (is (true? (-> convolution-info :args (nth 9) :optional?)))

      (is (= "no-bias" (-> convolution-info :args (nth 10) :name)))
      (is (= "boolean" (-> convolution-info :args (nth 10) :type)))
      (is (= "0" (-> convolution-info :args (nth 10) :default)))
      (is (true? (-> convolution-info :args (nth 10) :optional?)))

      (is (= "layout" (-> convolution-info :args (nth 13) :name)))
      (is (= "None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'" (-> convolution-info :args (nth 13) :type)))
      (is (= "'None'" (-> convolution-info :args (nth 13) :default)))
      (is (true? (-> convolution-info :args (nth 13) :optional?)))))

  (testing "element wise sum"
    (let [element-wise-sum-info (gen/gen-op-info "ElementWiseSum")]
      (is (= "add-n" (:fn-name element-wise-sum-info)))
      (is (= 1 (-> element-wise-sum-info :args count)))
      (is (= "num-args" (:key-var-num-args element-wise-sum-info)))

      (is (= "args" (-> element-wise-sum-info :args (nth 0) :name)))
      (is (= "NDArray-or-Symbol[]" (-> element-wise-sum-info :args (nth 0) :type)))
      (is (false? (-> element-wise-sum-info :args (nth 0) :optional?))))))

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
  (is (= ["foo" "bar" "bar-1"] (gen/rename-duplicate-params ["foo" "bar" "bar"])))
  (is (= ["foo" "bar" "bar-1" "foo-1"] (gen/rename-duplicate-params ["foo" "bar" "bar" "foo"])))
  (is (= ["foo" "bar" "bar-1" "bar-2"] (gen/rename-duplicate-params ["foo" "bar" "bar" "bar"])))
  (is (= ["foo" "bar" "bar-1" "bar-2" "foo-1" "baz"] (gen/rename-duplicate-params ["foo" "bar" "bar" "bar" "foo" "baz"]))))

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
  (is (= '(if (clojure.core/map? kwargs-map-or-vec-or-sym)
            (util/empty-list)
            (util/coerce-param
             kwargs-map-or-vec-or-sym
             #{"scala.collection.Seq"}))
         (gen/symbol-vector-args))))

(deftest test-symbol-map-args
  (is (= '(if (clojure.core/map? kwargs-map-or-vec-or-sym)
            (org.apache.clojure-mxnet.util/convert-symbol-map
             kwargs-map-or-vec-or-sym)
            nil)
         (gen/symbol-map-args))))

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
                 nil)))
           ar2))
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
                 nil)))
           ar3))))

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
    (is (= '(([sym sym-or-object]
              (util/coerce-return
               (.$div
                sym
                (util/nil-or-coerce-param
                 sym-or-object
                 #{"org.apache.mxnet.Symbol" "java.lang.Object"})))))
           (gen/gen-symbol-function-arity op-name op-values function-name)))))

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
                    #{"float" "org.apache.mxnet.NDArray"})))))
           (gen/gen-ndarray-function-arity op-name op-values)))))

(deftest test-write-to-file
  (testing "symbol-api"
    (let [fname "test/test-symbol-api.clj"
          fns (gen/all-symbol-api-functions gen/op-names)
          _ (gen/write-to-file [(first fns) (second fns)]
                               (gen/symbol-api-gen-ns false)
                               fname)]
      (is (= "activation"
             (file-function-name "test/good-test-symbol-api.clj")
             (file-function-name fname)))))

  (testing "symbol-random-api"
    (let [fname "test/test-symbol-random-api.clj"
          fns (gen/all-symbol-random-api-functions gen/op-names)
          _ (gen/write-to-file [(first fns) (second fns)]
                               (gen/symbol-api-gen-ns true)
                               fname)]
      (is (= "exponential"
             (file-function-name "test/good-test-symbol-random-api.clj")
             (file-function-name fname)))))


 (testing "symbol"
    (let [fname "test/test-symbol.clj"
          _ (gen/write-to-file [(first gen/all-symbol-functions)]
                               gen/symbol-gen-ns
                               fname)
          good-contents (slurp "test/good-test-symbol.clj")
          contents (slurp fname)]
      (is (= good-contents contents))))

  (testing "ndarray-api"
    (let [fname "test/test-ndarray-api.clj"
          fns (gen/all-ndarray-api-functions gen/op-names)
          _ (gen/write-to-file [(first fns) (second fns)]
                               (gen/ndarray-api-gen-ns false)
                               fname)]
      (is (= "activation"
             (file-function-name "test/good-test-ndarray-api.clj")
             (file-function-name fname)))))

  (testing "ndarray-random-api"
    (let [fname "test/test-ndarray-random-api.clj"
          fns (gen/all-ndarray-random-api-functions gen/op-names)
          _ (gen/write-to-file [(first fns) (second fns)]
                               (gen/ndarray-api-gen-ns true)
                               fname)]
      (is (= "exponential"
             (file-function-name "test/good-test-ndarray-random-api.clj")
             (file-function-name fname)))))

  (testing "ndarray"
    (let [fname "test/test-ndarray.clj"
          _ (gen/write-to-file [(first gen/all-ndarray-functions)]
                               gen/ndarray-gen-ns
                               fname)
          good-contents (slurp "test/good-test-ndarray.clj")
          contents (slurp fname)]
      (is (= good-contents contents)))))
