(ns test-helper
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]))

(def data-dir "test/test-images/")

(defn load-test-images []
  (when-not (.exists (io/file (str data-dir "Pug-Cookie.jpg")))
    (sh "./scripts/get_test_images.sh")))
