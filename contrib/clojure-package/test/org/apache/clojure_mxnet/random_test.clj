(ns org.apache.clojure-mxnet.random-test
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random]
            [clojure.test :refer :all]))

(deftest test-uniform-on-cpu
  (let [ctx (context/default-context)]
    (let [[a b] [-10 10]
          shape [100 100]
          _ (random/seed 128)
          un1 (random/uniform a b shape {:context ctx})
          _ (random/seed 128)
          un2 (random/uniform a b shape {:context ctx})]
      (is (= un1 un2))
      (is (<  (Math/abs
               (/ (/ (apply + (ndarray/->vec un1))
                     (- (ndarray/size un1) (+ a b)))
                  2.0))
              0.1)))))

(deftest test-normal-on-cpu
  (let [[mu sigma] [10 2]
        shape [100 100]
        _ (random/seed 128)
        ret1 (random/normal mu sigma shape)
        _ (random/seed 128)
        ret2 (random/normal mu sigma shape)]
    (is (= ret1 ret2))

    (let [array (ndarray/->vec ret1)
          mean (/ (apply + array) (count array))
          devs (map #(* (- % mean) (- % mean)) array)
          stddev (Math/sqrt (/ (apply + devs) (count array)))]
      (is (<  (Math/abs (- mean mu)) 0.1))
      (is (< (Math/abs (- stddev sigma)) 0.1)))))

