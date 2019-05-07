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



(ns bert.infer
  (:require [bert.util :as bert-util]
            [clojure.pprint :as pprint]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.util :as util]))

(def model-path-prefix "data/static_bert_qa")
;; epoch number of the model
(def epoch 2)
;; the maximum length of the sequence
(def seq-length 384)

;;; data helpers

(defn post-processing [result tokens]
  (let [output1 (ndarray/slice-axis result 2 0 1)
        output2 (ndarray/slice-axis result 2 1 2)
        ;;; get the formatted logits result
        start-logits (ndarray/reshape output1 [0 -3])
        end-logits (ndarray/reshape output2 [0 -3])
        start-prob (ndarray/softmax start-logits)
        end-prob (ndarray/softmax end-logits)
        start-idx (-> (ndarray/argmax start-prob 1)
                      (ndarray/->vec)
                      (first))
        end-idx (-> (ndarray/argmax end-prob 1)
                    (ndarray/->vec)
                    (first))]
    (if (> end-idx start-idx)
      (subvec tokens start-idx (inc end-idx))
      (subvec tokens end-idx (inc end-idx)))))

(defn make-predictor [ctx]
  (let [input-descs [{:name "data0"
                      :shape [1 seq-length]
                      :dtype dtype/FLOAT32
                      :layout layout/NT}
                     {:name "data1"
                      :shape [1 seq-length]
                      :dtype dtype/FLOAT32
                      :layout layout/NT}
                     {:name "data2"
                      :shape [1]
                      :dtype dtype/FLOAT32
                      :layout layout/N}]
        factory (infer/model-factory model-path-prefix input-descs)]
    (infer/create-predictor
     factory
     {:contexts [ctx]
      :epoch epoch})))

(defn pre-processing [ctx idx->token token->idx qa-map]
  (let [{:keys [input-question input-answer ground-truth-answers]} qa-map
       ;;; pre-processing tokenize sentence
        token-q (bert-util/tokenize (string/lower-case input-question))
        token-a (bert-util/tokenize (string/lower-case input-answer))
        valid-length (+ (count token-q) (count token-a))
        ;;; generate token types [0000...1111...0000]
        qa-embedded (into (bert-util/pad [] 0 (count token-q))
                          (bert-util/pad [] 1 (count token-a)))
        token-types (bert-util/pad qa-embedded 0 seq-length)
        ;;; make BERT pre-processing standard
        token-a (conj token-a "[SEP]")
        token-q (into [] (concat ["[CLS]"] token-q ["[SEP]"] token-a))
        tokens (bert-util/pad token-q "[PAD]" seq-length)
        ;;; pre-processing - token to index translation

        indexes (bert-util/tokens->idxs token->idx tokens)]
    {:input-batch [(ndarray/array indexes [1 seq-length] {:context ctx})
                   (ndarray/array token-types [1 seq-length] {:context ctx})
                   (ndarray/array [valid-length] [1] {:context ctx})]
     :tokens tokens
     :qa-map qa-map}))

(defn infer
  ([] (infer (context/default-context)))
  ([ctx]
   (let [predictor (make-predictor ctx)
         {:keys [idx->token token->idx]} (bert-util/get-vocab)
        ;;; samples taken from https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/
         question-answers (clojure.edn/read-string (slurp "squad-samples.edn"))]
     (doseq [qa-map question-answers]
       (let [{:keys [input-batch tokens qa-map]} (pre-processing ctx idx->token token->idx qa-map)
             result (first (infer/predict-with-ndarray predictor input-batch))
             answer (post-processing result tokens)]
         (println "===============================")
         (println "      Question Answer Data")
         (pprint/pprint qa-map)
         (println)
         (println "  Predicted Answer: " answer)
         (println "==============================="))))))

(defn -main [& args]
  (let [[dev] args]
    (if (= dev ":gpu")
      (infer (context/gpu))
      (infer (context/cpu)))))

(comment

  (infer)

  (infer (context/gpu))

  )
