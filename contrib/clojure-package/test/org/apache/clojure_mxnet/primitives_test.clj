(ns org.apache.clojure-mxnet.primitives-test
  (:require [org.apache.clojure-mxnet.primitives :as primitives]
            [clojure.test :refer :all])
  (:import (org.apache.mxnet MX_PRIMITIVES$MX_PRIMITIVE_TYPE
                             MX_PRIMITIVES$MX_FLOAT
                             MX_PRIMITIVES$MX_Double)))

(deftest test-primitive-types
  (is (not (primitives/primitive? 3)))
  (is (primitives/primitive? (primitives/mx-float 3)))
  (is (primitives/primitive? (primitives/mx-double 3))))

(deftest test-float-primitives
  (is (instance? MX_PRIMITIVES$MX_PRIMITIVE_TYPE (primitives/mx-float 3)))
  (is (instance? MX_PRIMITIVES$MX_FLOAT (primitives/mx-float 3)))
  (is (instance? Float (-> (primitives/mx-float 3)
                           (primitives/->num))))
  (is (= 3.0 (-> (primitives/mx-float 3)
                 (primitives/->num)))))

(deftest test-double-primitives
  (is (instance? MX_PRIMITIVES$MX_PRIMITIVE_TYPE (primitives/mx-double 2)))
  (is (instance? MX_PRIMITIVES$MX_Double (primitives/mx-double 2)))
  (is (instance? Double (-> (primitives/mx-double 2)
                            (primitives/->num))))
  (is (= 2.0 (-> (primitives/mx-double 2)
                 (primitives/->num)))))

