(ns org.apache.clojure-mxnet.resource-scope-test
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [clojure.test :refer :all]))

(deftest test-resource-scope-with-ndarray
  (let [x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (def temp-x (ndarray/ones [3 1]))
                    (def temp-y (ndarray/ones [3 1]))
                    (let [z {:just-a-test (def temp-z (ndarray/ones [3 3]))}
                          y (ndarray/+ temp-x 1)]
                      y))]
    (is (true? (.isDisposed temp-x)))
    (is (true? (.isDisposed temp-y)))
    (is (true? (.isDisposed temp-z)))
    (is (false? (.isDisposed return-val)))
    (is (false? (.isDisposed x)))
    (is (= [2.0 2.0 2.0] (ndarray/->vec return-val)))))

(deftest test-nested-resource-scope-with-ndarray
  (let [x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (def temp-x (ndarray/ones [3 1]))
                    (resource-scope/using
                     (def temp-y (ndarray/ones [3 1]))
                     (is (false? (.isDisposed temp-y)))
                     (is (false? (.isDisposed temp-x))))
                    (is (true? (.isDisposed temp-y)))
                    (is (false? (.isDisposed temp-x))))]
    (is (true? (.isDisposed temp-y)))
    (is (true? (.isDisposed temp-x)))
    (is (false? (.isDisposed x)))))

(deftest test-resource-scope-with-sym
  (let [x (sym/ones [2 2])
        return-val (resource-scope/using
                    (def temp-x (sym/ones [3 1]))
                    (def temp-y (sym/ones [3 1]))
                    (let [z {:just-a-test (def temp-z (sym/ones [3 3]))}
                          y (sym/+ temp-x 1)]
                      y))]
    (is (true? (.isDisposed temp-x)))
    (is (true? (.isDisposed temp-y)))
    (is (true? (.isDisposed temp-z)))
    (is (false? (.isDisposed return-val)))
    (is (false? (.isDisposed x)))))
