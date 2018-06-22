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
