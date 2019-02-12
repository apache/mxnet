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

(ns neural-style.vgg-19-test
	(:require 
		[clojure.test :refer :all]
		[opencv4.core :as cv]
		[clojure.java.io :as io]
		[org.apache.clojure-mxnet.ndarray :as ndarray]
		[org.apache.clojure-mxnet.context :as context]
		[neural-style.core :as neural]))

(defn pic-to-ndarray-vec[path]
	(-> path 
		cv/imread
	 	neural/image->ndarray))

(defn last-modified-check[x]
	(let [t (- (System/currentTimeMillis) (.lastModified x)) ]
	(if (> 10000 t) ; 10 seconds 
		x
		(throw (Exception. (str "Generated File Too Old: (" t " ms) [" x "]"))))))

(defn latest-pic-to-ndarray-vec[folder]
	 (->> folder 
	 	io/as-file
		(.listFiles)
		(sort-by #(.lastModified %))
		last
		(last-modified-check)
		(.getPath)
		pic-to-ndarray-vec))

(deftest vgg-19-test
	(neural/train [(context/cpu)] 3)
    (is (not (nil? (latest-pic-to-ndarray-vec "output")))))