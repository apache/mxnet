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

(ns infer.draw
  (:require
   [opencv4.colors.rgb :as rgb]
   [opencv4.core :refer [FONT_HERSHEY_PLAIN imread imwrite new-point put-text! rectangle]]))

(defn black-boxes! [img results]
  (doseq [{confidence :confidence label :label top-left :top-left bottom-right :bottom-right} results]
    (let [w (.width img)
          h (.height img)
          top-left-p (new-point (int (* w (first top-left))) (int (* h (second top-left))))
          bottom-right-p (new-point (int (* w (first bottom-right))) (int (* h (second bottom-right))))]
      (if (< 15 confidence)
        (do
          (rectangle img top-left-p bottom-right-p rgb/white 1)
          (put-text! img
                     (str label "[" confidence "% ]")
                     top-left-p
                     FONT_HERSHEY_PLAIN
                     1.0
                     rgb/white 1)))))
  img)

(defn draw-bounds [image results output-dir]
  (let [out-file (str output-dir "/" (.getName (clojure.java.io/as-file image)))]
    (-> image
        (imread)
        (black-boxes! results)
        (imwrite out-file))))