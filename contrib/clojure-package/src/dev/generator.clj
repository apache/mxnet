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
            [t6.from-scala.core :refer [$ $$] :as $]
            [clojure.reflect :as r]
            [clojure.pprint]
            [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet NDArray NDArrayAPI
                             Symbol SymbolAPI
                             Base Base$RefInt Base$RefLong Base$RefFloat Base$RefString)
           (scala.collection.mutable ListBuffer ArrayBuffer))
  (:gen-class))


(defn clojure-case
  "Transforms a scala string (function name) to clojure case"
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

(defn ndarray-transform-param-name [parameter-types]
  (transform-param-names util/ndarray-param-coerce parameter-types))

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
       (remove #(re-find #"org\$apache\$mxnet" (str (:name %))))
       (remove #(re-find #"\$default" (str (:name %))))))

(defn get-public-to-gen-methods [public-to-hand-gen public-no-default]
  (let [public-to-hand-gen-names
        (into #{} (mapv (comp str :name) public-to-hand-gen))]
    (remove #(-> % :name str public-to-hand-gen-names) public-no-default)))

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
    (let [fstr (-> f
                   clojure.pprint/pprint
                   with-out-str
                   (clojure.string/replace #"\\n\\n" "\n"))]
      (.write w fstr))
    (.write w "\n"))))

(defn remove-prefix
  [prefix s]
  (let [regex (re-pattern (str prefix "(.*)"))
        replacement "$1"]
  (clojure.string/replace s regex replacement)))

(defn in-namespace-random? [op-name]
  (or (clojure.string/includes? op-name "random_")
      (clojure.string/includes? op-name "sample_")))

(defn op-name->namespace-type [op-name]
  (cond
    (#{"uniform" "normal"} op-name)              :deprecated
    (clojure.string/includes? op-name "random_") :random
    (clojure.string/includes? op-name "sample_") :random
    :else                                        :core))

;;;;;;; Common operations

(def libinfo (Base/_LIB))

(def op-names
  (let [l ($ ListBuffer/empty)]
    (.mxListAllOpNames libinfo l)
    (->> l
         (util/buffer->vec)
         (remove #(or (= "Custom" %) (re-matches #"^_.*" %))))))

(defn- parse-arg-type [s]
  (let [[_ var-arg-type _ set-arg-type arg-spec _ type-req _ default-val] (re-find #"(([\w-\[\]\s]+)|\{([^}]+)\})\s*(\([^)]+\))?(,\s*(optional|required)(,\s*default=(.*))?)?" s)]
    {:type (clojure.string/trim (or set-arg-type var-arg-type))
     :spec arg-spec
     :optional? (or (= "optional" type-req)
                    (= "boolean" var-arg-type))
     :default default-val
     :orig s}))

(defn- get-op-handle [op-name]
  (let [ref (new Base$RefLong 0)]
    (do (.nnGetOpHandle libinfo op-name ref)
        (.value ref))))

(defn gen-op-info [op-name]
  (let [handle (get-op-handle op-name)
        name (new Base$RefString nil)
        desc (new Base$RefString nil)
        key-var-num-args (new Base$RefString nil)
        num-args (new Base$RefInt 0)
        arg-names ($ ListBuffer/empty)
        arg-types ($ ListBuffer/empty)
        arg-descs ($ ListBuffer/empty)]
    (do (.mxSymbolGetAtomicSymbolInfo libinfo
                                      handle
                                      name
                                      desc
                                      num-args
                                      arg-names
                                      arg-types
                                      arg-descs
                                      key-var-num-args)
        {:fn-name (clojure-case (.value name))
         :fn-description (.value desc)
         :args (mapv (fn [t n d] (assoc t :name n :description d))
                     (mapv parse-arg-type (util/buffer->vec arg-types))
                     (mapv clojure-case (util/buffer->vec arg-names))
                     (util/buffer->vec arg-descs))
         :key-var-num-args (clojure-case (.value key-var-num-args))})))

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
  (write-to-file all-symbol-functions
                 symbol-gen-ns
                 "src/org/apache/clojure_mxnet/gen/symbol.clj"))

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

(def ndarray-gen-ns
  "(ns org.apache.clojure-mxnet.ndarray
  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max
                            min repeat reverse set sort take to-array empty shuffle
                            ref])
  (:import (org.apache.mxnet NDArray Shape)))")


(defn generate-ndarray-file []
  (println "Generating ndarray file")
  (write-to-file all-ndarray-functions
                 ndarray-gen-ns
                 "src/org/apache/clojure_mxnet/gen/ndarray.clj"))

;;;;;;; SymbolAPI

(defn fn-name->random-fn-name
  [fn-name]
  (cond
    (clojure.string/starts-with? fn-name "-random-")
    (remove-prefix "-random-" fn-name)

    (clojure.string/starts-with? fn-name "-sample-")
    (str (remove-prefix "-sample-" fn-name) "-like")

    :else fn-name))

(defn symbol-api-coerce-param
  [{:keys [name sym type optional?]}]
  (let [coerced-param (case type
                        "Shape" `(when ~sym (~'mx-shape/->shape ~sym))
                        "NDArray-or-Symbol[]" `(~'clojure.core/into-array ~sym)
                        "Map[String, String]"
                        `(when ~sym
                           (->> ~sym
                                (mapv (fn [[~'k ~'v]] [~'k (str ~'v)]))
                                (into {})
                                ~'util/convert-map))
                        sym)
        nil-param-allowed? (#{"name" "attr"} name)]
    (if (and optional? (not nil-param-allowed?))
      `(~'util/->option ~coerced-param)
      coerced-param)))

(defn gen-symbol-api-doc [fn-description params]
  (let [param-descriptions (mapv (fn [{:keys [name description optional?]}]
                                   (str "`" name "`: "
                                        description
                                        (when optional? " (optional)")
                                        "\n"))
                                 params)]
    (str fn-description "\n\n"
         (apply str param-descriptions))))

(defn gen-symbol-api-default-arity [op-name params]
  (let [opt-params (filter :optional? params)
        coerced-params (mapv symbol-api-coerce-param params)
        default-args (array-map :keys (mapv :sym params)
                                :or (into {}
                                          (mapv (fn [{:keys [sym]}] [sym nil])
                                                opt-params))
                                :as 'opts)]
    `([~default-args]
      (~'util/coerce-return
       (~(symbol (str "SymbolAPI/" op-name))
        ~@coerced-params)))))

(defn symbol-api-gen-ns
  [random-namespace?]
  (str
    "(ns\n"
    "  ^{:doc \"Experimental\"}\n"
    (if random-namespace?
      "  org.apache.clojure-mxnet.symbol-random-api\n"
      "  org.apache.clojure-mxnet.symbol-api\n")
    "  (:refer-clojure :exclude [* - + > >= < <= / cast concat identity flatten load max\n"
    "                            min repeat reverse set sort take to-array empty sin\n"
    "                            get apply shuffle ref])\n"
    "  (:require [org.apache.clojure-mxnet.util :as util]\n"
    "            [org.apache.clojure-mxnet.shape :as mx-shape])\n"
    "  (:import (org.apache.mxnet SymbolAPI)))"))

(defn make-gen-symbol-api-function
  [{:keys [fn-name->fn-name] :or {fn-name->fn-name identity}}]
  (fn [op-name]
    (let [{:keys [fn-name fn-description args]}
          (-> op-name (gen-op-info) (update :fn-name fn-name->fn-name))
          params (mapv (fn [{:keys [name type optional?] :as opts}]
                         (assoc opts
                                :sym (symbol name)
                                :optional? (or optional?
                                               (= "NDArray-or-Symbol" type))))
                       (conj args
                             {:name "name"
                              :type "String"
                              :optional? true
                              :description "Name of the symbol"}
                             {:name "attr"
                              :type "Map[String, String]"
                              :optional? true
                              :description "Attributes of the symbol"}))
          doc (clojure.string/join
                "\n\n  "
                (-> (gen-symbol-api-doc fn-description params)
                    (clojure.string/split #"\n")))
          default-call (gen-symbol-api-default-arity op-name params)]
      `(~'defn ~(symbol fn-name)
         ~doc
         ~@default-call))))

(def gen-symbol-api-function
  (make-gen-symbol-api-function {}))

(def gen-symbol-random-api-function
  (make-gen-symbol-api-function {:fn-name->fn-name fn-name->random-fn-name}))

(defn all-symbol-api-functions [op-names]
  (->> op-names
       (filter #(= :core (op-name->namespace-type %)))
       (mapv gen-symbol-api-function)))

(count (all-symbol-api-functions op-names)) ;215

(defn all-symbol-random-api-functions [op-names]
  (->> op-names
       (filter #(= :random (op-name->namespace-type %)))
       (mapv gen-symbol-random-api-function)))

(count (all-symbol-random-api-functions op-names)) ;16

(defn generate-symbol-api-file [op-names]
  (println "Generating symbol-api file")
  (write-to-file (all-symbol-api-functions op-names)
                 (symbol-api-gen-ns false)
                 "src/org/apache/clojure_mxnet/gen/symbol_api.clj"))

(defn generate-symbol-random-api-file [op-names]
  (println "Generating symbol-random-api file")
  (write-to-file (all-symbol-random-api-functions op-names)
                 (symbol-api-gen-ns true)
                 "src/org/apache/clojure_mxnet/gen/symbol_random_api.clj"))

;;;;;;; NDArrayAPI

(defn ndarray-api-coerce-param
  [{:keys [sym type optional?]}]
  (let [coerced-param (case type
                        "Shape" `(when ~sym (~'mx-shape/->shape ~sym))
                        "NDArray-or-Symbol[]" `(~'clojure.core/into-array ~sym)
                        sym)]
    (if optional?
      `(~'util/->option ~coerced-param)
      coerced-param)))

(defn gen-ndarray-api-doc [fn-description params]
  (let [param-descriptions (mapv (fn [{:keys [name description optional?]}]
                                   (str "`" name "`: "
                                        description
                                        (when optional? " (optional)")
                                        "\n"))
                                 params)]
    (str fn-description "\n\n"
         (apply str param-descriptions))))

(defn gen-ndarray-api-default-arity [op-name params]
  (let [opt-params (filter :optional? params)
        coerced-params (mapv ndarray-api-coerce-param params)
        default-args (array-map :keys (mapv :sym params)
                                :or (into {}
                                          (mapv (fn [{:keys [sym]}] [sym nil])
                                                opt-params))
                                :as 'opts)]
    `([~default-args]
      (~'util/coerce-return
       (~(symbol (str "NDArrayAPI/" op-name))
        ~@coerced-params)))))

(defn gen-ndarray-api-required-arity [fn-name req-params]
  (let [req-args (->> req-params
                      (mapv (fn [{:keys [sym]}] [(keyword sym) sym]))
                      (into {}))]
    `(~(mapv :sym req-params)
      (~(symbol fn-name) ~req-args))))

(defn make-gen-ndarray-api-function
  [{:keys [fn-name->fn-name] :or {fn-name->fn-name identity}}]
  (fn [op-name]
    (let [{:keys [fn-name fn-description args]}
          (-> op-name (gen-op-info) (update :fn-name fn-name->fn-name))
          params (mapv (fn [{:keys [name] :as opts}]
                         (assoc opts :sym (symbol name)))
                       (conj args {:name "out"
                                   :type "NDArray-or-Symbol"
                                   :optional? true
                                   :description "Output array."}))
          doc (clojure.string/join
                "\n\n  "
                (-> (gen-ndarray-api-doc fn-description params)
                    (clojure.string/split #"\n")))
          opt-params (filter :optional? params)
          req-params (remove :optional? params)
          req-call (gen-ndarray-api-required-arity fn-name req-params)
          default-call (gen-ndarray-api-default-arity op-name params)]
      (if (= 1 (count req-params))
        `(~'defn ~(symbol fn-name)
           ~doc
           ~@default-call)
        `(~'defn ~(symbol fn-name)
           ~doc
           ~req-call
           ~default-call)))))

(def gen-ndarray-api-function
  (make-gen-ndarray-api-function {}))

(def gen-ndarray-random-api-function
  (make-gen-ndarray-api-function {:fn-name->fn-name fn-name->random-fn-name}))

(defn all-ndarray-api-functions [op-names]
  (->> op-names
       (filter #(= :core (op-name->namespace-type %)))
       (mapv gen-ndarray-api-function)))

(count (all-ndarray-api-functions op-names)) ; 213

(defn all-ndarray-random-api-functions [op-names]
  (->> op-names
       (filter #(= :random (op-name->namespace-type %)))
       (mapv gen-ndarray-random-api-function)))

(count (all-ndarray-random-api-functions op-names)) ;16

(defn ndarray-api-gen-ns [random-namespace?]
  (str
    "(ns\n"
    "  ^{:doc \"Experimental\"}\n"
    (if random-namespace?
      "  org.apache.clojure-mxnet.ndarray-random-api\n"
      "  org.apache.clojure-mxnet.ndarray-api\n")
    "  (:refer-clojure :exclude [* - + > >= < <= / cast concat flatten identity load max\n"
    "                            min repeat reverse set sort take to-array empty shuffle\n"
    "                            ref])\n"
    "  (:require [org.apache.clojure-mxnet.shape :as mx-shape]\n"
    "            [org.apache.clojure-mxnet.util :as util])\n"
    "  (:import (org.apache.mxnet NDArrayAPI)))"))

(defn generate-ndarray-api-file [op-names]
  (println "Generating ndarray-api file")
  (write-to-file (all-ndarray-api-functions op-names)
                 (ndarray-api-gen-ns false)
                 "src/org/apache/clojure_mxnet/gen/ndarray_api.clj"))

(defn generate-ndarray-random-api-file [op-names]
  (println "Generating ndarray-random-api file")
  (write-to-file (all-ndarray-random-api-functions op-names)
                 (ndarray-api-gen-ns true)
                 "src/org/apache/clojure_mxnet/gen/ndarray_random_api.clj"))


;;; autogen the files
(do
  (generate-ndarray-file)

  ;; NDArrayAPI
  (generate-ndarray-api-file op-names)
  (generate-ndarray-random-api-file op-names)

  (generate-symbol-file)

  ;; SymbolAPI
  (generate-symbol-api-file op-names)
  (generate-symbol-random-api-file op-names))


(comment

  (gen-op-info "ElementWiseSum")

  (gen-ndarray-api-function "Activation")

  (gen-symbol-api-function "Activation")

  (gen-ndarray-random-api-function "random_randint")

  (gen-ndarray-random-api-function "sample_normal")

  (gen-symbol-random-api-function "random_poisson")

  ;; This generates a file with the bulk of the nd-array functions
  (generate-ndarray-file)

  ;; This generates a file with the bulk of the symbol functions
  (generate-symbol-file))
