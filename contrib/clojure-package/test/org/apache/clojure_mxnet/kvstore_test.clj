(ns org.apache.clojure-mxnet.kvstore-test
  (:require [org.apache.clojure-mxnet.kvstore :as kvstore]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]))

(deftest test-init-and-pull
  (let [kv (kvstore/create)
        shape [2 1]
        out (ndarray/zeros shape)]
    (-> kv
        (kvstore/init "3" (ndarray/ones shape))
        (kvstore/pull "3" out))
    (is (= [1.0 1.0] (ndarray/->vec out)))))

(deftest test-push-and-pull
  (let [kv (kvstore/create)
        shape [2 1]
        out (ndarray/zeros shape)]
    (-> kv
        (kvstore/init "3" (ndarray/ones shape))
        (kvstore/push "3" (ndarray/* (ndarray/ones shape) 4))
        (kvstore/pull "3" out))
    (is (= [4.0 4.0] (ndarray/->vec out)))))

(deftest test-aggregate
  (let [shape [4 4]
        ks ["b" "c" "d"]
        kv (kvstore/create)
        num-devs 4
        devs (mapv (fn [_] (context/cpu)) (range num-devs))
        vals (mapv #(ndarray/ones shape {:ctx %}) devs)]
    (-> kv
        (kvstore/init "a" (ndarray/zeros shape))
        (kvstore/init ks [(ndarray/zeros shape) (ndarray/zeros shape) (ndarray/zeros shape)])
        (kvstore/push "a" vals)
        (kvstore/pull "a" vals))
    (is (= 0.0  (->> vals
                     (mapv ndarray/->vec)
                     flatten
                     (map #(- % num-devs))
                     (apply +))))
    (let [result (for [k ks]
                   (let [tmp-vals (mapv #(ndarray/* (ndarray/ones shape {:ctx %}) 2.0) devs)]
                     (-> kv
                         (kvstore/push k tmp-vals)
                         (kvstore/pull k tmp-vals))
                     (map ndarray/->vec tmp-vals)))]
      (is (= 0.0 (->> result
                      (flatten)
                      (map #(- % (* num-devs 2)))
                      (apply +)))))))

(deftest test-type
  (is (= "local" (-> (kvstore/create "local")
                     (kvstore/type)))))

(deftest test-get-numworkers
  (is (= 1 (-> (kvstore/create "local")
               (kvstore/num-workers)))))

(deftest test-get-rank
  (is (= 0 (-> (kvstore/create "local")
               (kvstore/rank)))))
