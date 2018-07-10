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

(ns org.apache.clojure-mxnet.context
  (:import (org.apache.mxnet Context)))

(defn cpu
  ([device-id]
   (new Context "cpu" device-id))
  ([]
   (cpu 0)))

(defn gpu
  ([device-id]
   (new Context "gpu" device-id))
  ([]
   (gpu 0)))

(defn cpu-context []
  (cpu))

(defn default-context [] (cpu-context))

(defn device-type [context]
  (.deviceType context))

(defn device-id [context]
  (.deviceId context))
