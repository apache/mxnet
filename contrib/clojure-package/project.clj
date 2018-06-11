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

(defproject org.apache.mxnet.contrib.clojure/clojure-mxnet "0.1.1-SNAPSHOT"
  :description "Clojure package for MXNet"
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [t6/from-scala "0.3.0"]
                 ;; Choose the right dependency for your system
                 [org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.2.0"]
                 ;[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "1.2.0"]
                 ;[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu "1.2.0"]
                 [org.clojure/tools.logging "0.4.0"]
                 [org.apache.logging.log4j/log4j-core "2.8.1"]
                 [org.apache.logging.log4j/log4j-api "2.8.1"]
                 [org.slf4j/slf4j-log4j12 "1.7.25" :exclusions [org.slf4j/slf4j-api]]]
  :pedantic? :skip
  :plugins [[lein-codox "0.10.3" :exclusions [org.clojure/clojure]]
            [lein-cloverage "1.0.10" :exclusions [org.clojure/clojure]]]
  :codox {:namespaces [#"^org\.apache\.clojure-mxnet\.(?!gen).*"]}
  :aliases {"generate-code" ["run" "-m" "dev.generator"]})
