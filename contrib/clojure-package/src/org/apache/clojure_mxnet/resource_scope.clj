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

(ns org.apache.clojure-mxnet.resource-scope
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet ResourceScope)))

(defmacro
  using
  "Uses a Resource Scope for all forms. This is a way to manage all Native Resources like NDArray and Symbol - it will deallocate all Native Resources by calling close on them automatically. It will not call close on Native Resources returned from the form.
  Example:
  (resource-scope/using
   (let [temp-x (ndarray/ones [3 1])
        temp-y (ndarray/ones [3 1])]
    (ndarray/+ temp-x temp-y))) "
  [& forms]
  `(ResourceScope/using (new ResourceScope) (util/forms->scala-fn ~@forms)))


(defmacro
  with-do
  "Alias for a do within a resource scope using.
  Example:
  (resource-scope/with-do
    (ndarray/ones [3 1])
    :all-cleaned-up)
  "
  [& forms]
  `(using (do ~@forms)))

(defmacro
  with-let
  "Alias for a let within a resource scope using.
  Example:
  (resource-scope/with-let [temp-x (ndarray/ones [3 1])
                            temp-y (ndarray/ones [3 1])]
  (ndarray/+ temp-x temp-y))"
  [& forms]
  `(using (let ~@forms)))
