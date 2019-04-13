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

(ns org.apache.clojure-mxnet.eval-metric-test
  (:require [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(defn test-eval-metric [test-metric metric-name labels preds metric-val]
  (println "Testing eval metric" metric-name)
  (let [metric test-metric]
    (eval-metric/update metric labels preds)
    (is (= [metric-name metric-val] (eval-metric/get metric)))

    (testing "get does not reset the metric"
      (is (= [metric-name metric-val] (eval-metric/get metric))))

    (testing "resetting the metric"
      (eval-metric/reset metric)
      (is (= [metric-name "NaN"] (map str (eval-metric/get metric)))))

    (testing "get-and-reset gets the metric and then resets it"
      (eval-metric/update metric labels preds)
      (is (= [metric-name metric-val] (eval-metric/get-and-reset metric)))
      (is (= [metric-name "NaN"] (map str (eval-metric/get metric)))))))

(deftest test-metrics
  (doseq [[metric-fn metric-name labels preds metric-val]
          [[(eval-metric/accuracy) "accuracy" [(ndarray/zeros [2])] [(ndarray/zeros [2 3])] 1.0]
           [(eval-metric/top-k-accuracy 2) "top_k_accuracy" [(ndarray/zeros [2])] [(ndarray/zeros [2 3])] 1.0]
           [(eval-metric/f1) "f1" [(ndarray/zeros [2])] [(ndarray/zeros [2 3])] 0.0]
           [(eval-metric/perplexity) "Perplexity" [(ndarray/ones [2])] [(ndarray/ones [2 3])] 1.0]
           [(eval-metric/mae) "mae" [(ndarray/ones [2])] [(ndarray/ones [2])] 0.0]
           [(eval-metric/mse) "mse" [(ndarray/ones [2])] [(ndarray/ones [2])] 0.0]
           [(eval-metric/rmse) "rmse" [(ndarray/ones [2])] [(ndarray/ones [2])] 0.0]]]
    (test-eval-metric metric-fn metric-name labels preds  metric-val)))

(deftest test-custom-metric
  (let [metric (eval-metric/custom-metric (fn [label pred]
                                            (float
                                             (- (apply + (ndarray/->vec label))
                                                (apply + (ndarray/->vec pred)))))
                                          "my-metric")]
    (eval-metric/update metric [(ndarray/ones [2])] [(ndarray/ones [2])])
    (is (= ["my-metric" 0.0] (eval-metric/get metric)))))

(deftest test-comp-metric
  (let [metric (eval-metric/comp-metric [(eval-metric/accuracy)
                                         (eval-metric/f1)
                                         (eval-metric/top-k-accuracy 2)])]
    (eval-metric/update metric [(ndarray/ones [2])] [(ndarray/ones [2 3])])
    (is (= {"accuracy" 0.0
            "f1" 0.0
            "top_k_accuracy" 1.0} (eval-metric/get metric)))))
