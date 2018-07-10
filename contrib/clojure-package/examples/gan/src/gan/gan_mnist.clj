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

(ns gan.gan-mnist
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as opt]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [gan.viz :as viz]
            [org.apache.clojure-mxnet.context :as context])
  (:gen-class))

;; based off of https://medium.com/@julsimon/generative-adversarial-networks-on-apache-mxnet-part-1-b6d39e6b5df1


(def data-dir "data/")
(def output-path "results/")
(def batch-size 100)
(def num-epoch 10)

(io/make-parents (str output-path "gout"))

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "../../scripts/get_mnist_data.sh"))

(defonce mnist-iter (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                       :label (str data-dir "train-labels-idx1-ubyte")
                                       :input-shape [1 28 28]
                                       :batch-size batch-size
                                       :shuffle true}))

(def rand-noise-iter (mx-io/rand-iter [batch-size 100 1 1]))

(comment

  ;;This is for figuring out the convolution and deconvolution layers to convert the image sizes

  (defn conv-output-size [input-size kernel-size padding stride]
    (float (inc (/ (- (+ input-size (* 2 padding)) kernel-size) stride))))

  ;; Calcing the layer sizes for discriminator
  (conv-output-size 28 4 3 2) ;=> 16
  (conv-output-size 16 4 1 2) ;=> 8
  (conv-output-size 8 4 1 2) ;=> 4.0
  (conv-output-size 4 4 0 1) ;=> 1

  ;; Calcing the layer sizes for generator
  (defn deconv-output-size [input-size kernel-size padding stride]
    (-
     (+ (* stride (- input-size 1))
        kernel-size)
     (* 2 padding)))

  (deconv-output-size 1 4 0 1) ;=> 4
  (deconv-output-size 4 4 1 2) ;=> 8
  (deconv-output-size 8 4 1 2) ;=> 16
  (deconv-output-size 16 4 3 2)) ;=> 28


(def ndf 28) ;; image height /width
(def nc 1) ;; number of channels
(def eps (float (+ 1e-5  1e-12)))
(def lr  0.0005) ;; learning rate
(def beta1 0.5)

(def label (sym/variable "label"))

(defn discriminator []
  (as-> (sym/variable "data") data
    (sym/convolution "d1" {:data data :kernel [4 4] :pad [3 3] :stride [2 2] :num-filter ndf :no-bias true})
    (sym/batch-norm "dbn1" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact1" {:data data :act-type "leaky" :slope 0.2})

    (sym/convolution "d2" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 2 ndf) :no-bias true})
    (sym/batch-norm "dbn2" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact1" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d3" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 3 ndf) :no-bias true})
    (sym/batch-norm "dbn3" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact3" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d4" {:data data :kernel [4 4] :pad [0 0] :stride [1 1] :num-filter (* 4 ndf) :no-bias true})
    (sym/flatten "flt" {:data data})

    (sym/fully-connected "fc" {:data data :num-hidden 1 :no-bias false})
    (sym/logistic-regression-output "dloss" {:data data :label label})))

(defn generator []
  (as-> (sym/variable "rand") data
    (sym/deconvolution "g1" {:data data :kernel [4 4]  :pad [0 0] :stride [1 1] :num-filter (* 4 ndf) :no-bias true})
    (sym/batch-norm "gbn1" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact1" {:data data :act-type "relu"})

    (sym/deconvolution "g2" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 2 ndf) :no-bias true})
    (sym/batch-norm "gbn2" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact2" {:data data :act-type "relu"})

    (sym/deconvolution "g3" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter ndf :no-bias true})
    (sym/batch-norm "gbn3" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact3" {:data data :act-type "relu"})

    (sym/deconvolution "g4" {:data data :kernel [4 4] :pad [3 3] :stride [2 2] :num-filter nc :no-bias true})
    (sym/activation "gact4" {:data data :act-type "tanh"})))

(let [data [(ndarray/ones [batch-size 100 1 1])]
      label [(ndarray/ones [batch-size 100 1 1])]]
  (def my-iter (mx-io/ndarray-iter data)))

(defn save-img-gout [i n x]
  (do
    (viz/im-sav {:title (str "gout-" i "-" n)
                 :output-path output-path
                 :x x
                 :flip false})))

(defn save-img-diff [i n x]
  (do (viz/im-sav {:title (str "diff-" i "-" n)
                   :output-path output-path
                   :x x
                   :flip false})))

(defn save-img-data [i n batch]
  (do (viz/im-sav {:title (str "data-" i "-" n)
                   :output-path output-path
                   :x (first (mx-io/batch-data batch))
                   :flip false})))

(defn calc-diff [i n diff-d]
  (let [diff (ndarray/copy diff-d)
        arr (ndarray/->vec diff)
        mean (/ (apply + arr) (count arr))
        std (let [tmp-a (map #(* (- % mean) (- % mean)) arr)]
              (float (Math/sqrt (/ (apply + tmp-a) (count tmp-a)))))]
    (let [calc-diff (ndarray/+ (ndarray/div (ndarray/- diff mean) std) 0.5)]

      (save-img-diff i n calc-diff))))

(defn train [devs]
  (let [mod-d  (-> (m/module (discriminator) {:contexts devs :data-names ["data"] :label-names ["label"]})
                   (m/bind {:data-shapes (mx-io/provide-data mnist-iter)
                            :label-shapes (mx-io/provide-label mnist-iter)
                            :inputs-need-grad true})
                   (m/init-params {:initializer (init/normal 0.02)})
                   (m/init-optimizer {:optimizer (opt/adam {:learning-rate lr :wd 0.0 :beta1 beta1})}))
        mod-g (-> (m/module (generator) {:contexts devs :data-names ["rand"] :label-names nil})
                  (m/bind {:data-shapes (mx-io/provide-data rand-noise-iter)})
                  (m/init-params {:initializer (init/normal 0.02)})
                  (m/init-optimizer {:optimizer (opt/adam {:learning-rate lr :wd 0.0 :beta1 beta1})}))]

    (println "Training for " num-epoch " epochs...")
    (doseq [i (range num-epoch)]
      (mx-io/reduce-batches mnist-iter
                            (fn [n batch]
                              (let [rbatch (mx-io/next rand-noise-iter)
                                    out-g (-> mod-g
                                              (m/forward rbatch)
                                              (m/outputs))
                                   ;; update the discriminiator on the fake
                                    grads-f  (mapv #(ndarray/copy (first %)) (-> mod-d
                                                                                 (m/forward {:data (first out-g) :label [(ndarray/zeros [batch-size])]})
                                                                                 (m/backward)
                                                                                 (m/grad-arrays)))
                                   ;; update the discrimintator on the real
                                    grads-r (-> mod-d
                                                (m/forward {:data (mx-io/batch-data batch) :label [(ndarray/ones [batch-size])]})
                                                (m/backward)
                                                (m/grad-arrays))
                                    _ (mapv (fn [real fake] (let [r (first real)]
                                                              (ndarray/set r (ndarray/+ r fake)))) grads-r grads-f)
                                    _ (m/update mod-d)
                                   ;; update the generator
                                    diff-d (-> mod-d
                                               (m/forward {:data (first out-g) :label [(ndarray/ones [batch-size])]})
                                               (m/backward)
                                               (m/input-grads))
                                    _ (-> mod-g
                                          (m/backward (first diff-d))
                                          (m/update))]
                                (when (zero? (mod n 100))
                                  (println "iteration = " i  "number = " n)
                                  (save-img-gout i n (ndarray/copy (ffirst out-g)))
                                  (save-img-data i n batch)
                                  (calc-diff i n (ffirst diff-d)))
                                (inc n)))))))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Running with context devices of" devs)
    (train devs)))

(comment
  (train [(context/cpu)]))
