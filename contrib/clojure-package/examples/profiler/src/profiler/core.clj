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

(ns profiler.core
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.profiler :as profiler]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.symbol :as sym])
  (:gen-class))

(def profiler-mode "symbolic") ;; can be symbolic, imperative, api, mem
(def output-path ".") ;; the profile file output directory
(def profiler-name "profile-matmul-20iter.json")
(def iter-num 100)
(def begin-profiling-iter 50)
(def end-profiling-iter 70)
(def gpu? false)

(defn run []
  (let [shape [4096 4096]
        path (str output-path "/" profiler-name)
        ctx (if gpu? (context/gpu) (context/cpu))
        kwargs {:filename path
                (keyword (str "profile-" profiler-mode)) 1}
        C (sym/dot "dot" [(sym/variable "A") (sym/variable "B")])
        a (random/uniform -1.0 1.0 shape {:ctx ctx})
        b (random/uniform -1.0 1.0 shape {:ctx ctx})
        exec (sym/bind C ctx {"A" [a] "B" [b]})]

    (profiler/profiler-set-config kwargs)
    (doseq [i (range iter-num)]
      (when (= i begin-profiling-iter)
        (profiler/profiler-set-state "run"))
      (when (= i end-profiling-iter)
        (profiler/profiler-set-state "stop"))
      (-> exec
          (executor/forward)
          (executor/outputs)
          (first)
          (ndarray/wait-to-read)))))

(defn -main [& args]
  (run))
