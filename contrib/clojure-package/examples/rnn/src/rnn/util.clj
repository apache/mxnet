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

(ns rnn.util
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]))

(defn build-vocab [path]
  (let [content (slurp path)
        vocab-map (reduce (fn [{:keys [vocab idx] :as result} c]
                            (if (get vocab c)
                              result
                              (-> result
                                  (update :vocab assoc c (inc idx))
                                  (update :idx inc))))
                          {:vocab {} :idx 0}     ;; 0 is used for padding
                          content)]
    (:vocab vocab-map)))

(defn make-revert-vocab [vmap]
  (into {} (map (fn [[k v]] [v k]) vmap)))

(defn make-input [char vocab arr]
  (let [idx (get vocab char)
        tmp (ndarray/zeros [1])]
    (do
      (ndarray/set tmp idx)
      (ndarray/set arr tmp))))

(defn cdf [weights]
  (let [total (* 1.0 (apply + weights))
        csums (reduce (fn [cumsum w] (conj cumsum (+ (or (last cumsum) 0) w))) [] weights)]
    (mapv #(/ % total) csums)))

(defn choice [population weights]
  (assert (= (count population) (count weights)))
  (let [cdf-vals (cdf weights)
        x (rand)
        idx (-> (partition-by (fn [v] (>= v x)) cdf-vals)
                first
                count)]
    (get population idx)))

;; we can use random output of fixed-output by choosing the largest probability
(defn make-output [prob fix-dict sample]
  (let [temperature 1.0
        char (if sample
               (let [scale-prob (mapv (fn [x] (if (< x 1e-6)
                                                1e-6
                                                (if (> x (- 1 1e-6))
                                                  (- 1 1e-6)
                                                  x))) prob)
                     rescale (mapv (fn [x] (Math/exp (/ (Math/log x) temperature))) scale-prob)
                     sum (apply + rescale)
                     rescale (map (fn [x] (/ x sum)) rescale)]
                 (choice fix-dict rescale))
               (->> (zipmap prob fix-dict)
                    (sort-by max)
                    (vals)
                    last))]
    char))
