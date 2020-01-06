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

(ns captcha.infer-ocr
  (:require [captcha.consts :refer :all]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(defn create-predictor
  []
  (let [data-desc {:name "data"
                   :shape [batch-size channels height width]
                   :layout layout/NCHW
                   :dtype dtype/FLOAT32}
        label-desc {:name "label"
                    :shape [batch-size label-width]
                    :layout layout/NT
                    :dtype dtype/FLOAT32}
        factory (infer/model-factory model-prefix
                                     [data-desc label-desc])]
    (infer/create-predictor factory)))

(defn -main
  [& args]
  (let [[filename] args
        image-fname (or filename "captcha_example.png")
        image-ndarray (-> image-fname
                          infer/load-image-from-file
                          (infer/reshape-image width height)
                          (infer/buffered-image-to-pixels [channels height width])
                          (ndarray/expand-dims 0))
        label-ndarray (ndarray/zeros [1 label-width])
        predictor (create-predictor)
        predictions (-> (infer/predict-with-ndarray
                         predictor
                         [image-ndarray label-ndarray])
                        first
                        (ndarray/argmax 1)
                        ndarray/->vec)]
    (println "CAPTCHA output:" (apply str (mapv int predictions)))))
