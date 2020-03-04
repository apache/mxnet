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

(defproject captcha "0.1.0-SNAPSHOT"
  :description "Captcha recognition via multi-label classification"
  :plugins [[lein-cljfmt "0.5.7"]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.7.0-SNAPSHOT"]]
  :main ^:skip-aot captcha.train-ocr
  :profiles {:train {:main captcha.train-ocr}
             :infer {:main captcha.infer-ocr}
             :uberjar {:aot :all}}
  :aliases {"train" ["with-profile" "train" "run"]
            "infer" ["with-profile" "infer" "run"]})
