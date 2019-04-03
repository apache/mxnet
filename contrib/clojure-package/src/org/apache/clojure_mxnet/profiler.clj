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

(ns org.apache.clojure-mxnet.profiler
  (:import (org.apache.mxnet Profiler))
  (:require [org.apache.clojure-mxnet.util :as util]))

(defn profiler-set-config
  " Set up the configure of profiler.
   -mode, optional Indicting whether to enable the profiler, can
    be symbolic or all. Default is symbolic.
   -fileName, optional The name of output trace file. Default is profile.json."
  [kwargs]
  (Profiler/profilerSetConfig
   (util/convert-io-map kwargs)))

(defn profiler-set-state
  "Set up the profiler state to record operator.
   -state, optional
   - Indicting whether to run the profiler, can
     be stop or run. Default is stop."
  ([state]
   (Profiler/profilerSetState state))
  ([]
   (profiler-set-state "stop")))

(defn dump-profile
  " Dump profile and stop profiler. Use this to save profile
   in advance in case your program cannot exit normally."
  ([finished]
   (Profiler/dumpProfile (int finished)))
  ([]
   (dump-profile 1)))
