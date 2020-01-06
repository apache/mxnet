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

(ns bert.util
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [cheshire.core :as json]))

(defn break-out-punctuation [s str-match]
  (->> (string/split (str s "<punc>") (re-pattern (str "\\" str-match)))
       (map #(string/replace % "<punc>" str-match))))

(defn break-out-punctuations [s]
  (if-let [target-char (first (re-seq #"[.,?!]" s))]
    (break-out-punctuation s target-char)
    [s]))

(defn tokenize [s]
  (->> (string/split s #"\s+")
       (mapcat break-out-punctuations)
       (into [])))

(defn pad [tokens pad-item num]
  (if (>= (count tokens) num)
    tokens
    (into tokens (repeat (- num (count tokens)) pad-item))))

(defn get-vocab []
  (let [vocab (json/parse-stream (io/reader "data/vocab.json"))]
    {:idx->token (get vocab "idx_to_token")
     :token->idx (get vocab "token_to_idx")}))

(defn tokens->idxs [token->idx tokens]
  (let [unk-idx (get token->idx "[UNK]")]
    (mapv #(get token->idx % unk-idx) tokens)))

(defn idxs->tokens [idx->token idxs]
  (mapv #(get idx->token %) idxs))
