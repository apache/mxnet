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

(defproject tutorial "0.1.0-SNAPSHOT"
  :description "MXNET tutorials"
  :plugins [[lein-cljfmt "0.5.7"]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 ;; Uncomment the one appropriate for your machine & configuration:
                 #_[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.3.0"]
                 #_[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu "1.3.0"]
                 #_[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "1.3.0"]])
