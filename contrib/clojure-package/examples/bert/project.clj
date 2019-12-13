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


(defproject bert "0.1.0-SNAPSHOT"
  :description "BERT Examples"
  :plugins [[lein-cljfmt "0.5.7"]
            ;;; lein-jupyter seems to have some incompatibilities with dependencies with cider
            ;;; so if you run into trouble please delete the `lein-juptyter` plugin
            [lein-jupyter "0.1.16" :exclusions [org.clojure/tools.nrepl org.clojure/clojure org.codehaus.plexus/plexus-utils org.clojure/tools.reader]]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.6.0-SNAPSHOT"]
                 [cheshire "5.8.1"]
                 [clojure-csv/clojure-csv "2.0.1"]]
  :pedantic? :skip
  :java-source-paths ["src/java"]
  :main bert.infer
  :repl-options {:init-ns bert.infer})
