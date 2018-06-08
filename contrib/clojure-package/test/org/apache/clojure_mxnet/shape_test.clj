(ns org.apache.clojure-mxnet.shape-test
  (:require [org.apache.clojure-mxnet.shape :as mx-shape]
            [clojure.test :refer :all]))

(deftest test-to-string
  (let [s (mx-shape/->shape [1 2 3])]
    (is (= "(1,2,3)" (str s)))))

(deftest test-equals
  (is (= (mx-shape/->shape [1 2 3]) (mx-shape/->shape [1 2 3])))
  (is (not= (mx-shape/->shape [1 2]) (mx-shape/->shape [1 2 3]))))
