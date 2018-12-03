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

(deftest test-optimizers-parameters-specs
  (is (thrown? Exception (optimizer/sgd {:wd 'a})))
  (is (thrown? Exception (optimizer/dcasgd {:lambda 'a})))
  (is (thrown? Exception (optimizer/nag {:momentum 'a})))
  (is (thrown? Exception (optimizer/ada-delta {:epsilon 'a})))
  (is (thrown? Exception (optimizer/rms-prop {:gamma1 'a})))
  (is (thrown? Exception (optimizer/ada-grad {:rescale-gradient 'a})))
  (is (thrown? Exception (optimizer/adam {:beta1 'a})))
  (is (thrown? Exception (optimizer/sgld {:lr-scheduler 0.1}))))