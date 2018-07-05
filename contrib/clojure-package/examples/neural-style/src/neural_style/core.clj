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

(ns neural-style.core
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.lr-scheduler :as lr-scheduler]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as opt]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.symbol :as sym]
            [mikera.image.core :as img]
            [mikera.image.filters :as img-filter]
            [think.image.pixel :as pixel]
            [neural-style.model-vgg-19 :as model-vgg-19])
  (:gen-class));; An Implementation of the paper A Neural Algorithm of Artistic Style
 ;;by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge

(def content-image "input/IMG_4343.jpg")
(def style-image "input/starry_night.jpg")
(def model-path "model/vgg19.params")
(def max-long-edge 600) ;; resize the content image
(def style-weight 1) ;; the weight for the style image
(def content-weight 5) ;; the weight for the content image
(def blur-radius 1) ;; the blur filter radius
(def output-dir "output")
(def lr 10) ;; the learning rate
(def tv-weight 0.01) ;; the magnitude on the tv loss
(def num-epochs 1000)
(def num-channels 3)

(defn image->ndarray [simg]
  (let [h (img/height simg)
        w (img/width simg)
        pixels (img/get-pixels simg)
        ;; normalize the pixels for vgg19
        rgb-pixels (reduce (fn [result pixel]
                             (let [[rs gs bs] result
                                   [r g b _] (pixel/unpack-pixel pixel)]
                               [(conj rs (- r 123.68))
                                (conj gs (- g 116.779))
                                (conj bs (- b 103.939))]))
                           [[] [] []]
                           pixels)]
    (println "The resized image is size " {:height h :width w})
    (-> rgb-pixels
        (flatten)
        (ndarray/array [1 num-channels h w]))))

(defn preprocess-content-image [path short-edge]
  (let [simg (img/load-image path)
        _ (println "The content image is size " {:height (img/height simg) :width (img/width simg)})
        factor (/ short-edge (img/width simg))
        resized-img (img/resize simg (* (img/width simg) factor) (* (img/height simg) factor))
        new-height (img/height resized-img)
        new-width (img/width resized-img)]
    (image->ndarray resized-img)))

(defn preprocess-style-image [path shape-vec]
  (let [[_ _ h w] shape-vec
        simg (img/load-image path)
        _ (println "The image is size " {:height (img/height simg) :width (img/width simg)})
        resized-img (img/resize simg w h)]
    (image->ndarray resized-img)))

(defn postprocess-image [img]
  (let [datas (ndarray/->vec img)
        image-shape (mx-shape/->vec (ndarray/shape img))
        spatial-size (* (get image-shape 2) (get image-shape 3))
        [rs gs bs] (doall (partition spatial-size datas))
        pixels  (mapv (fn [r g b]
                        (pixel/pack-pixel
                         (int (+ r 123.68))
                         (int (+ g 116.779))
                         (int (+ b 103.939))
                         (int 255)))
                      rs gs bs)
        new-image (img/new-image (get image-shape 3) (get image-shape 2))
        _  (img/set-pixels new-image (int-array pixels))]
    new-image))

(defn style-gram-symbol [input-size style]
  (let [[_ output-shape _] (sym/infer-shape style {:data [1 3 (first input-size) (second input-size)]})
        output-shapes (mx-shape/->vec output-shape)
        {:keys [gram-list grad-scale]} (doall (reduce
                                               (fn [result i]
                                                 (let [shape (get output-shapes i)
                                                       [s0 s1 s2 s3] shape
                                                       x (sym/reshape {:data (sym/get style i) :target-shape [s1 (* s2 s3)]})
                                                         ;; use fully connected to quickly do dot(x x^T)
                                                       gram (sym/fully-connected {:data x :weight x :no-bias true :num-hidden s1})]
                                                   (-> result
                                                       (update :gram-list conj gram)
                                                       (update :grad-scale conj (* s1 s2 s3 s1)))))
                                               {:gram-list [] :grad-scale []}
                                               (range (count (sym/list-outputs style)))))]
    {:gram (sym/group (into [] gram-list)) :g-scale grad-scale}))

(defn get-loss [gram content]
  (let [gram-loss (doall (mapv (fn [i]
                                 (let [gvar (sym/variable (str "target_gram_" i))]
                                   (sym/sum (sym/square (sym/- gvar (sym/get gram i))))))
                               (range (count (sym/list-outputs gram)))))
        cvar (sym/variable "target_content")
        content-loss (sym/sum (sym/square (sym/- cvar content)))]
    {:style-loss (sym/group gram-loss) :content-loss content-loss}))

(defn old-clip [v]
  (mapv (fn [a] (cond
                  (neg? a) 0
                  (> a 255) 255
                  :else a))
        v))

(defn clip [a]
  (cond
    (neg? a) 0
    (> a 255) 255
    :else a))

(defn save-image [img filename radius blur?]
  (let [filtered-image (if blur?
                         ((img-filter/box-blur blur-radius blur-radius) (postprocess-image img))
                         (postprocess-image img))]
    (do
      ;(img/show filtered-image) ;; Uncomment to have the image display 
      (img/write filtered-image filename "png"))))

(defn get-tv-grad-executor [img ctx tv-weight]
  (when (pos? tv-weight)
    (let [img-shape (mx-shape/->vec (ndarray/shape img))
          n-channel (get img-shape 1)
          s-img (sym/variable "img")
          s-kernel (sym/variable "kernel")
          channels (sym/split {:data s-img :axis 1 :num-outputs n-channel})
          out (sym/concat (doall (mapv (fn [i]
                                         (sym/convolution {:data (sym/get channels i) :weight s-kernel
                                                           :num-filter 1 :kernel [3 3] :pad [1 1] :no-bias true :stride [1 1]}))
                                       (range n-channel))))
          kernel (ndarray/* (ndarray/array [0 -1 0 -1 4 -1 0 -1 0] [1 1 3 3] {:ctx ctx})
                            0.8)
          out (ndarray/* out tv-weight)]
      (sym/bind out ctx {"img" img "kernel" kernel}))))

(defn train [devs]

  (let [dev (first devs)
        content-np (preprocess-content-image content-image max-long-edge)
        content-np-shape (mx-shape/->vec (ndarray/shape content-np))
        style-np (preprocess-style-image style-image content-np-shape)
        size [(get content-np-shape 2) (get content-np-shape 3)]
        {:keys [style content]} (model-vgg-19/get-symbol)
        {:keys [gram g-scale]} (style-gram-symbol size style)
        model-executor (model-vgg-19/get-executor gram content model-path size dev)

        _ (ndarray/set (:data model-executor) style-np)
        _ (executor/forward (:executor model-executor))

        style-array (mapv #(ndarray/copy %) (:style model-executor))

        mode-executor nil
        _ (ndarray/set (:data model-executor) content-np)
        _ (executor/forward (:executor model-executor))
        content-array (ndarray/copy (:content model-executor))

        {:keys [style-loss content-loss]} (get-loss gram content)
        model-executor (model-vgg-19/get-executor style-loss content-loss model-path size dev)

        grad-array (-> (doall (mapv (fn [i]
                                      (do
                                        (ndarray/set  (get (:arg-map model-executor) (str "target_gram_" i)) (get style-array i))
                                        (ndarray/* (ndarray/ones [1] {:ctx dev}) (/ style-weight (get g-scale i)))))
                                    (range (count style-array))))
                       (conj (ndarray/* (ndarray/ones [1] {:ctx dev}) content-weight)))

        _ (ndarray/copy-to content-array (get (:arg-map model-executor) "target_content"))

        ;;;train

        ;;initialize with random noise
        img (ndarray/- (random/uniform 0 255 content-np-shape dev) 128)
        ;;; img (random/uniform -0.1 0.1 content-np-shape dev)
        ;; img content-np
        lr-sched (lr-scheduler/factor-scheduler 10 0.9)

        _ (save-image content-np (str output-dir "/input.png") blur-radius false)
        _ (save-image style-np (str output-dir "/style.png") blur-radius false)

        optimizer (opt/adam {:learning-rate lr
                             :wd 0.005
                             :lr-scheduler lr-sched})
        optim-state (opt/create-state optimizer 0 img)

        _ (println "Starting training....")
        old-img (ndarray/copy-to img dev)
        clip-norm (apply  * (mx-shape/->vec (ndarray/shape img)))
        tv-grad-executor (get-tv-grad-executor img dev tv-weight)
        eps 0.0
        e 0]
    (doseq [i (range 20)]
      (ndarray/set (:data model-executor) img)
      (-> (:executor model-executor)
          (executor/forward)
          (executor/backward grad-array))

      (let [g-norm (ndarray/to-scalar (ndarray/norm (:data-grad model-executor)))]
        (if (> g-norm clip-norm)
          (ndarray/set (:data-grad model-executor) (ndarray/* (:data-grad model-executor) (/ clip-norm g-norm)))))

      (if tv-grad-executor
        (do
          (executor/forward tv-grad-executor)
          (opt/update optimizer 0
                      img
                      (ndarray/+ (:data-grad model-executor) (first (executor/outputs tv-grad-executor)))
                      optim-state))
        (opt/update optimizer 0 img (:data-grad model-executor) optim-state))

      (let [eps (ndarray/to-scalar
                 (ndarray/div (ndarray/norm (ndarray/- old-img img))
                              (ndarray/norm img)))]
        (println "Epoch " i  "relative change " eps)
        (when (zero? (mod i 2))
          (save-image (ndarray/copy img) (str output-dir "/out_" i ".png") blur-radius true)))

      (ndarray/set old-img img))))

(defn -main [& args]
  ;;; Note this only works on cpu right now
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Running with context devices of" devs)
    (train devs)))

(comment

  (train [(context/cpu)]))
