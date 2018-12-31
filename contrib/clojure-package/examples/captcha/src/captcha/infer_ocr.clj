(ns captcha.infer-ocr
  (:require [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(def batch-size 8)
(def channels 3)
(def height 30)
(def width 80)
(def label-width 4)
(def model-prefix "ocr")

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
