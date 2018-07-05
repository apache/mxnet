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

(ns org.apache.clojure-mxnet.kvstore
  (:refer-clojure :exclude [type])
  (:require [clojure.spec.alpha :as spec]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.spec.alpha :as s])
  (:import (org.apache.mxnet KVStore NDArray)))

(defn create
  " Create a new KVStore
   WARNING: it is your responsibility to clear this object through dispose.
   - name : #{local, dist} (default is local)
        The type of KVStore
        - local works for multiple devices on a single machine (single process)
        - dist works for multi-machines (multiple processes)"
  ([name]
   (KVStore/create name))
  ([]
   (create "local")))

(defn dispose
  "Release the native memory.
   The object shall never be used after it is disposed."
  [kvstore]
  (.dispose kvstore))

(s/def ::ks (s/or :string string?
                  :vec-of-string (s/coll-of string? :kind vector?)))
(s/def ::ndarray #(instance? NDArray %))
(s/def ::vs (s/or :ndarray ::ndarray
                  :vec-of-ndarray (s/coll-of ::ndarray :kind vector?)))

(defn init
  "Initialize a single or a sequence of key-value pairs into the store.
    For each key, one must init it before push and pull.
    Only worker 0's (rank == 0) data are used.
    This function returns after data have been initialized successfully
    kvstore - KVstore
    ks - keys (vec or strings or single string)
    vs - values (vec or NDArrays or single ndarry)"
  [kvstore ks vs]
  (util/validate! ::ks ks "Invalid keys")
  (util/validate! ::vs vs "Invalid values")
  (doto kvstore
    (.init (into-array (if (vector? ks) ks [ks]))
           (into-array (if (vector? vs) vs [vs])))))

(s/def ::priority int?)

(defn push
  " Push a single or a sequence of key-value pairs into the store.
    Data consistency:
    1. this function returns after adding an operator to the engine.
    2. push is always called after all previous push and pull on the same key are finished
    3. there is no synchronization between workers. One can use _barrier() to sync all workers

   -ks Keys
   -vs values  According values
   - priority
           The priority of the push operation.
           The higher the priority, the faster this action is likely
           to be executed before other push actions."
  ([kvstore ks vs priority]
   (util/validate! ::ks ks "Invalid keys")
   (util/validate! ::vs vs "Invalid values")
   (util/validate! ::priority priority "Invalid priority")
   (let [store-vals (if (vector? vs) vs [vs])
         store-keys (if (vector? ks) ks (into [] (repeat (count store-vals) ks)))]
     (doto kvstore
       (.push (into-array store-keys)
              (into-array store-vals)
              (int priority)))))
  ([kvstore ks vs]
   (push kvstore ks vs 0)))

(s/def ::outs (s/or :ndarray ::ndarray
                    :vec-of-ndarray (s/coll-of ::ndarray :kind vector?)))

(defn pull
  " Pull a single value or a sequence of values from the store.
    Data consistency:
    1. this function returns after adding an operator to the engine. But any
       further read on out will be blocked until it is finished.
    2. pull is always called after all previous push and pull on the same key are finished
    3. It pulls the newest value from the store.
   - kvstore
   - ks single or vector of (strings)
   - outs single or vector of outs (NDArrays)
   - priority
       The priority of the push operation.
       The higher the priority, the faster this action is likely
       to be executed before other push actions."
  ([kvstore ks outs priority]
   (util/validate! ::ks ks "Invalid keys")
   (util/validate! ::outs outs "Invalid outs")
   (util/validate! ::priority priority "Invalid priority")
   (let [store-vals (if (vector? outs) outs [outs])
         store-keys (if (vector? ks) ks (into [] (repeat (count store-vals) ks)))]
     (doto kvstore
       (.pull (into-array store-keys)
              (into-array store-vals)
              (int priority)))))
  ([kvstore ks outs]
   (pull kvstore ks outs 0)))

(defn type
  "Get the type of the kvstore"
  [kvstore]
  (.type kvstore))

(defn num-workers
  "Get the number of worker nodes"
  [kvstore]
  (.numWorkers kvstore))

(defn rank
  "Get the rank of this worker node
   returns The rank of this node, which is in [0, get_num_workers()) "
  [kvstore]
  (.rank kvstore))

(defn set-optimizer
  "Register an optimizer to the store
   If there are multiple machines, this process (should be a worker node)
   will pack this optimizer and send it to all servers. It returns after
   this action is done"
  [kvstore optimizer]
  (doto kvstore
    (.setOptimizer optimizer)))

(defn barrier
  "Global barrier among all worker nodes
    For example, assume there are n machines, we want to let machine 0 first
    init the values, and then pull the inited value to all machines. Before
    pulling, we can place a barrier to guarantee that the initialization is
    finished."
  [kvstore]
  (doto kvstore
    (.barrier kvstore)))

(defn num-dead-node [kvstore node-id]
  (.numDeadNode kvstore (int node-id)))

(defn set-barrier-before-exit
  " Whether to do barrier when the kvstore finalizes
   - kvstore
   - barrier-before-exit boolean"
  [kvstore barrier-before-exit]
  (doto kvstore
    (.setBarrierBeforeExit barrier-before-exit)))

(s/def ::head int?)
(s/def ::body string?)

(defn send-command-to-servers
  "Send a command to all server nodes
    Send a command to all server nodes, which will make each server node run
    KVStoreServer.controller
    This function returns after the command has been executed in all server nodes
   -kvstore
   -head the head of the command
   - body the body of the command"
  [kvstore head body]
  (util/validate! ::head head "Invalid head")
  (util/validate! ::body body "Invalid body")
  (doto kvstore
    (.sendCommandToServers (int head) body)))

(s/def ::fname string?)

(defn save-optimizer-states
  "Save optimizer (updater) state to file
   - kvstore
   - fname Path to output states file."
  [kvstore fname]
  (util/validate! ::fname fname "Invalid filename")
  (doto kvstore
    (.saveOptimizerStates fname)))

(defn load-optimizer-states
  "Load optimizer (updater) state from file
   - kvstore
   -fname Path to input states file."
  [kvstore fname]
  (util/validate! ::fname fname "Invalid filename")
  (doto kvstore
    (.loadOptimizerStates fname)))
