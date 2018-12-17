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

(ns imclassification.train-mnist-test
	(:require 
		[clojure.test :refer :all]
		[clojure.java.io :as io]
		[clojure.string :as s]
		[org.apache.clojure-mxnet.context :as context]
		[org.apache.clojure-mxnet.module :as module]
		[imclassification.train-mnist :as mnist]))

(defn- file-to-filtered-seq [file]
	(->>
		file 
		(io/file)
		(io/reader)
		(line-seq)
		(filter  #(not (s/includes? % "mxnet_version")))))

(deftest mnist-two-epochs-test
	(module/save-checkpoint (mnist/start [(context/cpu)] 2) {:prefix "target/test" :epoch 2})
	(is (= 
		(file-to-filtered-seq "test/test-symbol.json.ref") 
		(file-to-filtered-seq "target/test-symbol.json"))))