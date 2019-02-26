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

(ns org.apache.clojure-mxnet.primitives-test
  (:require [org.apache.clojure-mxnet.primitives :as primitives]
            [clojure.test :refer :all])
  (:import (org.apache.mxnet MX_PRIMITIVES$MX_PRIMITIVE_TYPE
                             MX_PRIMITIVES$MX_FLOAT
                             MX_PRIMITIVES$MX_Double)))

(deftest test-primitive-types
  (is (not (primitives/primitive? 3)))
  (is (primitives/primitive? (primitives/mx-float 3)))
  (is (primitives/primitive? (primitives/mx-double 3))))

(deftest test-float-primitives
  (is (instance? MX_PRIMITIVES$MX_PRIMITIVE_TYPE (primitives/mx-float 3)))
  (is (instance? MX_PRIMITIVES$MX_FLOAT (primitives/mx-float 3)))
  (is (instance? Float (-> (primitives/mx-float 3)
                           (primitives/->num))))
  (is (= 3.0 (-> (primitives/mx-float 3)
                 (primitives/->num)))))

(deftest test-double-primitives
  (is (instance? MX_PRIMITIVES$MX_PRIMITIVE_TYPE (primitives/mx-double 2)))
  (is (instance? MX_PRIMITIVES$MX_Double (primitives/mx-double 2)))
  (is (instance? Double (-> (primitives/mx-double 2)
                            (primitives/->num))))
  (is (= 2.0 (-> (primitives/mx-double 2)
                 (primitives/->num)))))

