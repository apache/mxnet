(defproject bert-qa "0.1.0-SNAPSHOT"
  :description "BERT QA Example"
  :plugins [[lein-cljfmt "0.5.7"]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.5.0-SNAPSHOT"]
                 [cheshire "5.8.1"]]
  :pedantic? :skip
  :java-source-paths ["src/java"]
  :main bert-qa.core
  :repl-options {:init-ns bert-qa.core})
