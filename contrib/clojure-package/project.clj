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

(defproject org.apache.mxnet.contrib.clojure/clojure-mxnet "1.6.0-SNAPSHOT"
  :description "Clojure package for MXNet"
  :url "https://github.com/apache/incubator-mxnet"
  :license {:name "Apache License"
            :url "http://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [t6/from-scala "0.3.0"]

                 ;; To use with nightly snapshot
                 ;[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "<insert-snapshot-version>"]
                 ;[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "<insert-snapshot-version>"]
                 ;[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu "<insert-snapshot-version"]

                 ;;; CI / Local Build
                 [org.apache.mxnet/mxnet-full_2.11 "INTERNAL"]

                 [org.clojure/tools.logging "0.4.0"]
                 [org.apache.logging.log4j/log4j-core "2.8.1"]
                 [org.apache.logging.log4j/log4j-api "2.8.1"]
                 [org.slf4j/slf4j-log4j12 "1.7.25" :exclusions [org.slf4j/slf4j-api]]]
  :pedantic? :skip
  :plugins [[lein-codox "0.10.6" :exclusions [org.clojure/clojure]]
            [lein-cloverage "1.0.10" :exclusions [org.clojure/clojure]]
            [lein-cljfmt "0.5.7"]]
  :codox {:namespaces [#"^org\.apache\.clojure-mxnet\.(?!gen).*"]}
  :aot [dev.generator]
  :repositories [["staging" {:url "https://repository.apache.org/content/repositories/staging"                  :snapshots true
                             :update :always}]
                 ["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"               :snapshots true
                               :update :always}]])
