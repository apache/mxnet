(ns org.apache.clojure-mxnet.callback-test
  (:require [org.apache.clojure-mxnet.callback :as callback]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(deftest test-speedometer
  (let [speedometer (callback/speedometer 1)
        metric (eval-metric/accuracy)]
    (eval-metric/update metric [(ndarray/ones [2])] [(ndarray/ones [2 3])])
    ;;; only side effects of logging
    (callback/invoke speedometer 0 1 metric)
    (callback/invoke speedometer 0 2 metric)
    (callback/invoke speedometer 0 3 metric)
    (callback/invoke speedometer 0 10 metric)
    (callback/invoke speedometer 0 50 metric)
    (callback/invoke speedometer 0 100 metric)))
