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

(ns dev.generator
  (:require [t6.from-scala.core :as scala]
            [clojure.reflect :as r]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.pprint])
  (:import (org.apache.mxnet NDArray NDArrayAPI
                             Symbol SymbolAPI))
  (:gen-class))


(defn clojure-case
  [string]
  (-> string
      (clojure.string/replace #"(\s+)([A-Z][a-z]+)" "$1-$2")
      (clojure.string/replace #"([A-Z]+)([A-Z][a-z]+)" "$1-$2")
      (clojure.string/replace #"([a-z0-9])([A-Z])" "$1-$2")
      (clojure.string/lower-case)
      (clojure.string/replace #"\_" "-")
      (clojure.string/replace #"\/" "div")))

(defn transform-param-names [coerce-fn parameter-types]
  (->> parameter-types
       (map str)
       (map (fn [x] (or (coerce-fn x) x)))
       (map (fn [x] (last (clojure.string/split x #"\."))))))

(defn symbol-transform-param-name [parameter-types]
  (transform-param-names util/symbol-param-coerce parameter-types))

(defn symbol-api-transform-param-name [parameter-types]
  (transform-param-names util/symbol-api-param-coerce parameter-types))

(defn ndarray-transform-param-name [parameter-types]
  (transform-param-names util/ndarray-param-coerce parameter-types))

(defn ndarray-api-transform-param-name [parameter-types]
  (transform-param-names util/ndarray-api-param-coerce parameter-types))

(defn has-variadic? [params]
  (->> params
       (map str)
       (filter (fn [s] (re-find #"\&" s)))
       count
       pos?))


(defn increment-param-name [pname]
  (if-let [num-str (re-find #"-\d" pname)]
    (str 
     (first (clojure.string/split pname #"-"))
     "-"
     (inc (Integer/parseInt (last (clojure.string/split num-str #"-")))))
    (str pname "-" 1)))

(defn rename-duplicate-params [pnames]
  (->> (reduce
        (fn [pname-counts n]
          (let [rn (if (pname-counts n) (str n "-" (pname-counts n)) n)
                inc-pname-counts (update-in pname-counts [n] (fnil inc 0))]
            (update-in inc-pname-counts [:params] conj rn)))
        {:params []}
        pnames)
       :params))

(defn get-public-no-default-methods [obj]
  (->> (r/reflect obj)
       :members
       (map #(into {} %))
       (filter #(-> % :flags :public))
       (filter #(not (re-find #"org\$apache\$mxnet" (str (:name %)))))
       (filter #(not (re-find #"\$default" (str (:name %)))))))

(defn get-public-to-gen-methods [public-to-hand-gen public-no-default]
  (remove #(contains? (->> public-to-hand-gen
                           (mapv :name)
                           (mapv str)
                           (set))
                      (str (:name %)))
          public-no-default))

(defn public-by-name-and-param-count [public-reflect-info]
 (->> public-reflect-info
      (group-by :name)
      (map (fn [[k v]] [k (group-by #(count (:parameter-types %)) v)]))
      (into {})))

(def license
  (str
   ";; Licensed to the Apache Software Foundation (ASF) under one or more\n"
   ";; contributor license agreements.  See the NOTICE file distributed with\n"
   ";; this work for additional information regarding copyright ownership.\n"
   ";; The ASF licenses this file to You under the Apache License, Version 2.0\n"
   ";; (the \"License\"); you may not use this file except in compliance with\n"
   ";; the License.  You may obtain a copy of the License at\n"
   ";;\n"
   ";;    http://www.apache.org/licenses/LICENSE-2.0\n"
   ";;\n"
   ";; Unless required by applicable law or agreed to in writing, software\n"
   ";; distributed under the License is distributed on an \"AS IS\" BASIS,\n"
   ";; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
   ";; See the License for the specific language governing permissions and\n"
   ";; limitations under the License.\n"
   ";;\n"))

(defn write-to-file [functions ns-gen fname]
  (with-open [w (clojure.java.io/writer fname)]
    (.write w ns-gen)
    (.write w "\n\n")
    (.write w ";; Do not edit - this is auto-generated")
    (.write w "\n\n")
    (.write w license)
    (.write w "\n\n")
    (.write w "\n\n")
  (doseq [f functions]
    (clojure.pprint/pprint f w)
    (.write w "\n"))))

;;;;;;; Symbol

(def symbol-public-no-default
  (get-public-no-default-methods Symbol))

(into #{} (mapcat :parameter-types symbol-public-no-default))
;; #{java.lang.Object scala.collection.Seq scala.Option long double scala.collection.immutable.Map int ml.dmlc.mxnet.Executor float ml.dmlc.mxnet.Context java.lang.String scala.Enumeration$Value ml.dmlc.mxnet.Symbol int<> ml.dmlc.mxnet.Symbol<> ml.dmlc.mxnet.Shape java.lang.String<>}

(def symbol-hand-gen-set
  #{"scala.Option"
    "scala.Enumeration$Value"
    "org.apache.mxnet.Context"
    "scala.Tuple2"
    "scala.collection.Traversable"})

;;; min and max have a conflicting arity of 2 with the auto gen signatures
(def symbol-filter-name-set #{"max" "min"})

(defn is-symbol-hand-gen? [info]
  (or
   (->> (:name info)
        str
        (get symbol-filter-name-set))
   (->> (map str (:parameter-types info))
        (into #{})
        (clojure.set/intersection symbol-hand-gen-set)
        count
        pos?)))

(def symbol-public-to-hand-gen
  (filter is-symbol-hand-gen? symbol-public-no-default))
(def symbol-public-to-gen
  (get-public-to-gen-methods symbol-public-to-hand-gen
                             symbol-public-no-default))


(count symbol-public-to-hand-gen) ;=> 35 mostly bind!
(count symbol-public-to-gen) ;=> 307

(into #{} (map :name symbol-public-to-hand-gen))
;;=>  #{arange bind ones zeros simpleBind Variable}



(defn symbol-vector-args []
  `(if (map? ~'kwargs-map-or-vec-or-sym)
     (~'util/empty-list)
     (~'util/coerce-param ~'kwargs-map-or-vec-or-sym #{"scala.collection.Seq"})))

(defn symbol-map-args []
  `(if (map? ~'kwargs-map-or-vec-or-sym)
     (util/convert-symbol-map ~'kwargs-map-or-vec-or-sym)
     nil))


(defn add-symbol-arities [params function-name]
  (if (= ["sym-name" "kwargs-map" "symbol-list" "kwargs-map-1"]
         (mapv str params))
    [`([~'sym-name ~'attr-map ~'kwargs-map]
       (~function-name ~'sym-name (~'util/convert-symbol-map ~'attr-map) (~'util/empty-list) (~'util/convert-symbol-map ~'kwargs-map)))
     `([~'sym-name ~'kwargs-map-or-vec-or-sym]
       (~function-name ~'sym-name nil ~(symbol-vector-args) ~(symbol-map-args)))
     `([~'kwargs-map-or-vec-or-sym]
       (~function-name nil nil  ~(symbol-vector-args) ~(symbol-map-args)))]))

(defn gen-symbol-function-arity [op-name op-values function-name]
  (mapcat
   (fn [[param-count info]]
     (let [targets (->> (mapv :parameter-types info)
                        (apply interleave)
                        (mapv str)
                        (partition (count info))
                        (mapv set))
           pnames (->> (mapv :parameter-types info)
                       (mapv symbol-transform-param-name)
                       (apply interleave)
                       (partition (count info))
                       (mapv #(clojure.string/join "-or-" %))
                       (rename-duplicate-params)
                       (mapv symbol))
           coerced-params (mapv (fn [p t] `(~'util/nil-or-coerce-param ~(symbol (clojure.string/replace p #"\& " "")) ~t)) pnames targets)
           params (if (= #{:public :static} (:flags (first info)))
                    pnames
                    (into ['sym] pnames))
           function-body (if (= #{:public :static} (:flags (first info)))
                           `(~'util/coerce-return (~(symbol (str "Symbol/" op-name)) ~@coerced-params))
                           `(~'util/coerce-return (~(symbol (str  "." op-name)) ~'sym ~@coerced-params)
                             ))]
       (when (not (and (> param-count 1) (has-variadic? params)))
         `[(
            ~params
            ~function-body
            )
           ~@(add-symbol-arities params function-name)])))
   op-values))


(def all-symbol-functions
 (for [operation  (sort (public-by-name-and-param-count symbol-public-to-gen))]
   (let [[op-name op-values] operation
         function-name (-> op-name
                           str
                           scala/decode-scala-symbol
                           clojure-case
                           symbol)]
     `(~'defn ~function-name
       ~@(remove nil? (gen-symbol-function-arity op-name op-values function-name))))))



(def symbol-gen-ns "(ns org.apache.clojure-mxnet.symbol
  (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max
                            min repeat reverse set sort take to-array empty sin
                            get apply shuffle ref])
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet Symbol)))")


(defn generate-symbol-file []
  (println "Generating symbol file")
  (write-to-file all-symbol-functions symbol-gen-ns "src/org/apache/clojure_mxnet/gen/symbol.clj"))

;;;;;;; SymbolAPI

(def symbol-api-public-no-default
  (get-public-no-default-methods SymbolAPI))

(into #{} (mapcat :parameter-types symbol-api-public-no-default))

(def symbol-api-hand-gen-set
  #{"scala.Enumeration$Value"
    "scala.Tuple2"
    "scala.collection.mutable.Map"
    "scala.collection.Traversable"})

(defn is-symbol-api-hand-gen? [info]
  (->> (map str (:parameter-types info))
       (into #{})
       (clojure.set/intersection symbol-api-hand-gen-set)
       count
       pos?))

(def symbol-api-public-to-hand-gen
  (filter is-symbol-api-hand-gen? symbol-api-public-no-default))
(def symbol-api-public-to-gen
  (get-public-to-gen-methods symbol-api-public-to-hand-gen
                             symbol-api-public-no-default))

(count symbol-api-public-to-hand-gen)   ;=> 1 (Custom)
(count symbol-api-public-to-gen)        ;=> 232

(into #{} (map :name symbol-api-public-to-hand-gen))
;;=>  #{Custom}

(defn symbol-api-reflect-info [name]
  (->> symbol-api-public-no-default
       (filter #(= name (str (:name %))))
       first))

(def activation-sym (symbol-api-reflect-info "Activation"))

(defn gen-symbol-api-function-arity [op-name op-values function-name]
  (mapcat
   (fn [[param-count info]]
     (let [targets (->> (mapv :parameter-types info)
                        (apply interleave)
                        (mapv str)
                        (partition (count info))
                        (mapv set))
           pnames (->> (mapv :parameter-types info)
                       (mapv symbol-api-transform-param-name)
                       (apply interleave)
                       (partition (count info))
                       (mapv #(clojure.string/join "-or-" %))
                       (rename-duplicate-params)
                       (mapv symbol))
           coerced-params (mapv (fn [p t]
                                  (if (= #{"scala.Option"} t)
                                    `(~'util/nil-or-coerce-param (~'util/->option ~(symbol (clojure.string/replace p #"\& " ""))) ~t)
                                    `(~'util/nil-or-coerce-param ~(symbol (clojure.string/replace p #"\& " "")) ~t)))
                                pnames targets)
           params (if (= #{:public :static} (:flags (first info)))
                    pnames
                    (into ['sym] pnames))
           function-body (if (= #{:public :static} (:flags (first info)))
                           `(~'util/coerce-return (~(symbol (str "SymbolAPI/" op-name)) ~@coerced-params))
                           `(~'util/coerce-return (~(symbol (str  "." op-name)) ~'sym ~@coerced-params)
                             ))]
       (when (not (and (> param-count 1) (has-variadic? params)))
         `[(
            ~params
            ~function-body
            )])))
   op-values))

(defn gen-symbol-api-functions [public-to-gen-methods]
 (for [operation (sort (public-by-name-and-param-count public-to-gen-methods))]
   (let [[op-name op-values] operation
         function-name (-> op-name
                           str
                           scala/decode-scala-symbol
                           clojure-case
                           symbol)]
     `(~'defn ~function-name
       ~@(remove nil? (gen-symbol-api-function-arity op-name op-values function-name))))))

(gen-symbol-api-functions [activation-sym])

(def all-symbol-api-functions
  (gen-symbol-api-functions symbol-api-public-to-gen))

(def symbol-api-gen-ns "(ns org.apache.clojure-mxnet.symbol-api
  (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max
                            min repeat reverse set sort take to-array empty sin
                            get apply shuffle ref])
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet SymbolAPI)))")

(defn generate-symbol-api-file []
  (println "Generating symbol-api file")
  (write-to-file all-symbol-api-functions symbol-api-gen-ns "src/org/apache/clojure_mxnet/gen/symbol_api.clj"))

;;;;;;; NDArray


(def ndarray-public-no-default
  (get-public-no-default-methods NDArray))

(def ndarray-hand-gen-set
  #{"org.apache.mxnet.NDArrayFuncReturn"
    "org.apache.mxnet.Context"
    "scala.Enumeration$Value"
    "scala.Tuple2"
    "scala.collection.Traversable"})

(defn is-ndarray-hand-gen? [info]
  (->> (map str (:parameter-types info))
       (into #{})
       (clojure.set/intersection ndarray-hand-gen-set)
       count
       pos?))


(def ndarray-public-to-hand-gen
  (filter is-ndarray-hand-gen? ndarray-public-no-default))
(def ndarray-public-to-gen
  (get-public-to-gen-methods ndarray-public-to-hand-gen
                             ndarray-public-no-default))


(count ndarray-public-to-hand-gen) ;=> 15
(count ndarray-public-to-gen) ;=> 486

(->> ndarray-public-to-hand-gen (map :name) (into #{}))



(defn gen-ndarray-function-arity [op-name op-values]
  (for [[param-count info] op-values]
    (let [targets (->> (mapv :parameter-types info)
                       (apply interleave)
                       (mapv str)
                       (partition (count info))
                       (mapv set))
          pnames (->> (mapv :parameter-types info)
                      (mapv ndarray-transform-param-name)
                      (apply interleave)
                      (partition (count info))
                      (mapv #(clojure.string/join "-or-" %))
                      (rename-duplicate-params)
                      (mapv symbol))
          coerced-params (mapv (fn [p t] `(~'util/coerce-param ~(symbol (clojure.string/replace p #"\& " "")) ~t)) pnames targets)
          params (if (= #{:public :static} (:flags (first info)))
                   pnames
                   (into ['ndarray] pnames))
          function-body (if (= #{:public :static} (:flags (first info)))
                          `(~'util/coerce-return (~(symbol (str "NDArray/" op-name)) ~@coerced-params))
                          `(~'util/coerce-return (~(symbol (str  "." op-name)) ~'ndarray ~@coerced-params)
                            ))]
      (when (not (and (> param-count 1) (has-variadic? params)))
        `(
          ~params
          ~function-body
          )))))


(defn gen-ndarray-functions [public-to-gen-methods]
  (for [operation (sort (public-by-name-and-param-count public-to-gen-methods))]
    (let [[op-name op-values] operation
          function-name (-> op-name
                            str
                            scala/decode-scala-symbol
                            clojure-case
                            symbol)]
      `(~'defn ~function-name
        ~@(remove nil? (gen-ndarray-function-arity op-name op-values))))))

(def all-ndarray-functions
  (gen-ndarray-functions ndarray-public-to-gen))

(def ndarray-gen-ns "(ns org.apache.clojure-mxnet.ndarray
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:import (org.apache.mxnet NDArray Shape)))")


(defn generate-ndarray-file []
  (println "Generating ndarray file")
  (write-to-file all-ndarray-functions
                 ndarray-gen-ns
                 "src/org/apache/clojure_mxnet/gen/ndarray.clj"))

;;;;;;; NDArrayAPI

(def ndarray-api-public-no-default
  (get-public-no-default-methods NDArrayAPI))

(def ndarray-api-hand-gen-set
  #{"org.apache.mxnet.NDArrayFuncReturn"
    "org.apache.mxnet.Context"
    "scala.Enumeration$Value"
    "scala.Tuple2"
    "scala.collection.Traversable"})

(defn is-ndarray-api-hand-gen? [info]
  (->> (map str (:parameter-types info))
       (into #{})
       (clojure.set/intersection ndarray-api-hand-gen-set)
       count
       pos?))


(def ndarray-api-public-to-hand-gen
  (filter is-ndarray-api-hand-gen? ndarray-api-public-no-default))
(def ndarray-api-public-to-gen
  (get-public-to-gen-methods ndarray-api-public-to-hand-gen
                             ndarray-api-public-no-default))


(count ndarray-api-public-to-hand-gen)   ; => 0
(count ndarray-api-public-to-gen)        ; => 232

(map :name ndarray-api-public-to-gen)

(defn ndarray-api-reflect-info [name]
  (->> ndarray-api-public-no-default
       (filter #(= name (str (:name %))))
       first))

(def activation (ndarray-api-reflect-info "Activation"))
(def batch-norm (ndarray-api-reflect-info "BatchNorm"))


(defn add-ndarray-api-arities [params function-name]
  (let [req-params (->> params
                        reverse
                        (drop-while #(re-find #"[O|o]ption" (str %)))
                        reverse)
        num-req (count req-params)
        num-options (- (count params) num-req)]
    (for [i (range num-options)]
      `([~@(take (+ num-req i) params)]
        (~function-name
         ~@(take (+ num-req i) params)
         ~@(repeat (- num-options i) 'util/none))))))

(defn gen-ndarray-api-function-arity [op-name op-values function-name]
  (mapcat
   (fn [[param-count info]]
     (let [targets (->> (mapv :parameter-types info)
                       (apply interleave)
                       (mapv str)
                       (partition (count info))
                       (mapv set))
          pnames (->> (mapv :parameter-types info)
                      (mapv ndarray-api-transform-param-name)
                      (apply interleave)
                      (partition (count info))
                      (mapv #(clojure.string/join "-or-" %))
                      (rename-duplicate-params)
                      (mapv symbol))
          coerced-params (mapv (fn [p t] `(~'util/coerce-param ~(symbol (clojure.string/replace p #"\& " "")) ~t)) pnames targets)
          params (if (= #{:public :static} (:flags (first info)))
                   pnames
                   (into ['ndarray] pnames))
          function-body (if (= #{:public :static} (:flags (first info)))
                          `(~'util/coerce-return (~(symbol (str "NDArrayAPI/" op-name)) ~@coerced-params))
                          `(~'util/coerce-return (~(symbol (str  "." op-name)) ~'ndarray ~@coerced-params)
                            ))]
      (when (not (and (> param-count 1) (has-variadic? params)))
        `[(
            ~params
            ~function-body
           )
          ~@(add-ndarray-api-arities params function-name)])))
   op-values))


(defn gen-ndarray-api-functions [public-to-gen-methods]
  (for [operation (sort (public-by-name-and-param-count public-to-gen-methods))]
    (let [[op-name op-values] operation
          function-name (-> op-name
                            str
                            scala/decode-scala-symbol
                            clojure-case
                            symbol)]
      `(~'defn ~function-name
        ~@(remove nil? (gen-ndarray-api-function-arity op-name op-values function-name))))))

(gen-ndarray-api-functions [activation])

(def all-ndarray-api-functions
  (gen-ndarray-api-functions ndarray-api-public-to-gen))

(def ndarray-api-gen-ns "(ns org.apache.clojure-mxnet.ndarray-api
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet NDArrayAPI)))")


(defn generate-ndarray-api-file []
  (println "Generating ndarray-api file")
  (write-to-file all-ndarray-api-functions
                 ndarray-api-gen-ns
                 "src/org/apache/clojure_mxnet/gen/ndarray_api.clj"))

;;; autogen the files
(do
  (generate-ndarray-file)
  (generate-ndarray-api-file)
  (generate-symbol-file)
  (generate-symbol-api-file))


(comment

  ;; This generates a file with the bulk of the nd-array functions
  (generate-ndarray-file)

  ;; This generates a file with the bulk of the symbol functions
  (generate-symbol-file)  )
