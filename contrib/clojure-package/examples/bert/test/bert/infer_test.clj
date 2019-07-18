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


(ns bert.infer-test
  (:require [bert.infer :refer :all]
            [bert.util :as util]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.infer :as infer]))

(def model-dir "data/")

(when-not (.exists (io/file (str model-dir "static_bert_qa-0002.params")))
  (println "Downloading bert qa data")
  (sh "./get_bert_data.sh"))

(deftest infer-test
  (let [ctx (context/default-context)
        predictor (make-predictor ctx)
        {:keys [idx->token token->idx]} (util/get-vocab)
        ;;; samples taken from https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/
        question-answers (clojure.edn/read-string (slurp "squad-samples.edn"))]
    (let [qa-map (last question-answers)
          {:keys [input-batch tokens qa-map]} (pre-processing ctx idx->token token->idx qa-map)
          result (first (infer/predict-with-ndarray predictor input-batch))]
      (is (= ["rich" "hickey"] (post-processing result tokens))))))
