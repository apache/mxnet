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

(ns org.apache.clojure-mxnet.callback
  (:import (org.apache.mxnet Callback$Speedometer)))

;;; used to track status during epoch

(defn speedometer
  ([batch-size frequent]
   (new Callback$Speedometer (int batch-size) (int frequent)))
  ([batch-size]
   (speedometer batch-size 50)))

(defn invoke [callback epoch nbatch metric]
  (doto callback
    (.invoke (int epoch) (int nbatch) metric)))
