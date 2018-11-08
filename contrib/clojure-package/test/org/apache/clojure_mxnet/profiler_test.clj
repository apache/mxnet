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

(ns org.apache.clojure-mxnet.profiler-test
  (:require [org.apache.clojure-mxnet.profiler :as profiler]
            [clojure.test :refer :all]))

;; Just excercising the interop

(deftest test-profiler
  (do
    (profiler/profiler-set-config  {:filename "test-profile.json"
                                    :profile-symbolic 1})
    (profiler/profiler-set-state "run")
    (profiler/profiler-set-state "stop")
    (profiler/profiler-set-state)
    (profiler/dump-profile 0)))
