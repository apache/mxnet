(ns org.apache.clojure-mxnet.test-util
  (:require [clojure.test :as t]))

(defn approx= [tolerance x y]
  (if (and (number? x) (number? y))
    (let [diff (Math/abs (- x y))]
      (< diff tolerance))
    (reduce (fn [x y] (and x y))
            (map #(approx= tolerance %1 %2) x y))))

