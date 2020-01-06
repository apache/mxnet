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

(ns org.apache.clojure-mxnet.initializer
  (:refer-clojure :exclude [apply])
  (:import (org.apache.mxnet Uniform Normal Xavier)))

(defn uniform
  "Initialize the weight with uniform [-scale, scale]
   scale - The scale of uniform distribution"
  ([scale]
   (new Uniform (float scale)))
  ([]
   (uniform 0.07)))

(defn normal
  "Initialize the weight with normal(0, sigma)
   sigma -  Standard deviation for gaussian distribution."
  ([sigma]
   (new Normal (float sigma)))
  ([]
   (normal 0.01)))

(defn xavier
  "Initialize the weight with Xavier or similar initialization scheme
  rand-type - 'gaussian' or 'uniform'
  factor-type - 'avg' 'in' or 'out'
  magnitude - scale of random number range "
  ([{:keys [rand-type factor-type magnitude :as opts]
     :or {rand-type "uniform"
          factor-type "avg"
          magnitude 3}}]
   (new Xavier rand-type factor-type (float magnitude)))
  ([]
   (xavier {})))

(defn apply [initializer name arr]
  (let [r (.apply initializer name arr)]
    arr))

(defn init-weight [initializer name arr]
  (doto initializer
    (.initWeight name arr)))
