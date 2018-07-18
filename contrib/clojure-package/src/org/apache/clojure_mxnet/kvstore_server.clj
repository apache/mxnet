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

(ns org.apache.clojure-mxnet.kvstore-server
  (:require [clojure.spec.alpha :as spec]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet KVStoreServer)))

(s/def ::env-map (s/map-of string? string?))

(defn init [env-map]
  (util/validate! ::env-map env-map "Invalid environment map")
  (KVStoreServer/init (util/convert-map env-map)))

(s/def ::die-if-others-go-out-timeout int?)

(defn start
  ([die-if-others-go-out-timeout]
   (util/validate! ::die-if-others-go-out-timeout die-if-others-go-out-timeout "Invalid setting")
   (KVStoreServer/start die-if-others-go-out-timeout))
  ([]
   (start 0)))
