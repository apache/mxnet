(ns org.apache.clojure-mxnet.optimizer-test
  (:require [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [clojure.test :refer :all]))

(defn test-optimizer [[opt-name optimizer-fn]]
  (println "Testing optimizer - " opt-name)
  (let [s (sym/variable "data")
        s (sym/fully-connected {:data s :num-hidden 100})
        ;; single device
        mod (m/module s {:data-names ["data"] :label-names nil})]
    (-> mod
        (m/bind {:data-shapes [{:name "data" :shape [10 10] :layout "NT"}]})
        (m/init-params)
        (m/init-optimizer {:optimizer (optimizer-fn)})
        (m/update))))


(deftest test-optimizer-update
  (let [opts [["sgd" optimizer/sgd]
              ["dcasgd" optimizer/dcasgd]
              ["nag" optimizer/nag]
              ["ada-delta" optimizer/ada-delta]
              ["rms-prop" optimizer/rms-prop]
              ["ada-grad" optimizer/ada-grad]
              ["adam" optimizer/adam]
              ["sgld" optimizer/sgld]]]
    (doseq [opt opts]
      (test-optimizer opt))))
