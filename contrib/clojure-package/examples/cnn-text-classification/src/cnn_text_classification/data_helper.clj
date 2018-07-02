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

(ns cnn-text-classification.data-helper
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.random :as random])
  (:import (java.io DataInputStream))
  (:gen-class))

(def w2v-file-path "../../data/GoogleNews-vectors-negative300.bin") ;; the word2vec file path
(def max-vectors 100) ;; If you are using word2vec embeddings and you want to only load part of them

(defn r-string [dis]
  (let [max-size 50
        bs (byte-array max-size)
        sb (new StringBuilder)]
    (loop [b (.readByte dis)
           i 0]
      (if (and (not= 32 b) (not= 10 b))
        (do (aset bs i b)
            (if (= 49 i)
              (do (.append sb (new String bs))
                  (recur (.readByte dis) 0))
              (recur (.readByte dis) (inc i))))
        (.append sb (new String bs 0 i))))
    (.toString sb)))

(defn get-float [b]
  (-> 0
      (bit-or (bit-shift-left (bit-and (aget b 0) 0xff) 0))
      (bit-or (bit-shift-left (bit-and (aget b 1) 0xff) 8))
      (bit-or (bit-shift-left (bit-and (aget b 2) 0xff) 16))
      (bit-or (bit-shift-left (bit-and (aget b 3) 0xff) 24))))

(defn read-float [is]
  (let [bs (byte-array 4)]
    (do (.read is bs)
        (get-float bs))))

(defn load-google-model [path]
  (println "Loading the word2vec model from binary ...")
  (with-open [bis (io/input-stream path)
              dis (new DataInputStream bis)]
    (let [word-size (Integer/parseInt (r-string dis))
          dim  (Integer/parseInt (r-string dis))
          _  (println "Processing with " {:dim dim :word-size word-size} " loading max vectors " max-vectors)
          word2vec (reduce (fn [r _]
                             (assoc r (r-string dis)
                                    (mapv (fn [_] (read-float dis)) (range dim))))
                           {}
                           (range max-vectors))]
      (println "Finished")
      {:num-embed dim :word2vec word2vec})))

(defn clean-str [s]
  (-> s
      (string/replace #"^A-Za-z0-9(),!?'`]" " ")
      (string/replace #"'s" " 's")
      (string/replace #"'ve" " 've")
      (string/replace #"n't" " n't")
      (string/replace #"'re" " 're")
      (string/replace #"'d" " 'd")
      (string/replace #"'ll" " 'll")
      (string/replace #"," " , ")
      (string/replace #"!" " ! ")
      (string/replace #"\(" " ( ")
      (string/replace #"\)" " ) ")
      (string/replace #"\?" " ? ")
      (string/replace #" {2,}" " ")
      (string/trim)));; Loads MR polarity data from files, splits the data into words and generates labels.
 ;; Returns split sentences and labels.
(defn load-mr-data-and-labels [path max-examples]
  (println "Loading all the movie reviews from " path)
  (let [positive-examples (mapv #(string/trim %) (-> (slurp (str path "/rt-polarity.pos"))
                                                     (string/split #"\n")))
        negative-examples (mapv #(string/trim %) (-> (slurp (str path "/rt-polarity.neg"))
                                                     (string/split #"\n")))
        positive-examples (into [] (if max-examples (take max-examples positive-examples) positive-examples))
        negative-examples (into [] (if max-examples (take max-examples negative-examples) negative-examples))
        ;; split by words
        x-text (->> (into positive-examples negative-examples)
                    (mapv clean-str)
                    (mapv #(string/split % #" ")))

        ;; generate labels
        positive-labels (mapv (constantly 1) positive-examples)
        negative-labels (mapv (constantly 0) negative-examples)]
    {:sentences x-text :labels (into positive-labels negative-labels)}))

;; Pads all sentences to the same length. The length is defined by the longest sentence.
;; Returns padded sentences.
(defn pad-sentences [sentences]
  (let [padding-word "<s>"
        sequence-len (apply max (mapv count sentences))]
    (mapv (fn [s] (let [diff (- sequence-len (count s))]
                    (if (pos? diff)
                      (into s (repeat diff padding-word))
                      s)))
          sentences)));; Map sentences and labels to vectors based on a pretrained embeddings
(defn build-input-data-with-embeddings [sentences embedding-size embeddings]
  (mapv (fn [sent]
          (mapv (fn [word] (or (get embeddings word)
                               (ndarray/->vec (random/uniform -0.25 0.25 [embedding-size]))))
                sent))
        sentences))

(defn load-ms-with-embeddings [path embedding-size embeddings max-examples]
  (println "Translating the movie review words into the embeddings")
  (let [{:keys [sentences labels]} (load-mr-data-and-labels path max-examples)
        sentences-padded  (pad-sentences sentences)
        data (build-input-data-with-embeddings sentences-padded embedding-size embeddings)]
    {:data data
     :label labels
     :sentence-count (count data)
     :sentence-size (count (first data))
     :embedding-size embedding-size}))

(defn read-text-embedding-pairs [rdr]
  (for [^String line (line-seq rdr)
        :let [fields (.split line " ")]]
    [(aget fields 0)
     (mapv #(Double/parseDouble ^String %) (rest fields))]))

(defn load-glove [glove-file-path]
  (println "Loading the glove pre-trained word embeddings from " glove-file-path)
  (into {} (read-text-embedding-pairs (io/reader glove-file-path))))

