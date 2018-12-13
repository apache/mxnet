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

(ns rnn.core_test
 (:require 
 	[rnn.test-char-rnn :as rnn]
 	[clojure.test :refer :all]))

(deftest check-trained-network
	(is (= 
		"The joke that we can start by the challenges of the American people. The American people have been talking about how to compete with the streets of San Antonio who the courage to come together as one "
	 (rnn/rnn-test "data/obama" 75 200 false))))