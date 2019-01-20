(ns org.apache.clojure-mxnet.resource-scope
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import (org.apache.mxnet ResourceScope)))

(defmacro using
  "Uses a Resource Scope for all forms. This is a way to manage all Native Resources like NDArray and Symbol - it will deallocate all Native Resources by calling close on them automatically. It will not call close on Native Resources returned from the form"
  [& forms]
  `(ResourceScope/using (new ResourceScope) (util/forms->scala-fn ~@forms)))
