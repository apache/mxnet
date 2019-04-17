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

(ns org.apache.clojure-mxnet.module
  "Module API for Clojure package."
  (:refer-clojure :exclude [update symbol])
  (:require [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [clojure.java.io :as io]
            [clojure.spec.alpha :as s]
            [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:import (org.apache.mxnet.module Module FitParams BaseModule)
           (org.apache.mxnet.io MXDataIter NDArrayIter)
           (org.apache.mxnet Initializer Optimizer NDArray DataBatch
                             Context EvalMetric Monitor Callback$Speedometer
                             DataDesc)))

(defn module
  "Module is a basic module that wrap a `symbol`.
    `sym`: Symbol definition.
    `opts-map` {
      `data-names`: vector of strings - Default is [\"data\"]
          Input data names
      `label-names`: vector of strings - Default is [\"softmax_label\"]
          Input label names
      `contexts`: Context - Default is `context/cpu`.
      `workload-list`: Default nil
          Indicating uniform workload.
      `fixed-param-names`: Default nil
          Indicating no network parameters are fixed.
    }
   Ex:
     (module sym)
     (module sym {:data-names [\"data\"]
                  :label-names [\"linear_regression_label\"]}"
  ([sym {:keys [data-names label-names contexts
                workload-list fixed-param-names] :as opts
         :or {data-names ["data"]
              label-names ["softmax_label"]
              contexts [(context/default-context)]}}]
   (new Module
        sym
        (util/vec->indexed-seq data-names)
        (util/vec->indexed-seq label-names)
        (into-array contexts)
        (util/->option (when workload-list (util/vec->indexed-seq workload-list)))
        (util/->option (when fixed-param-names (util/vec->set fixed-param-names)))))
  ([sym data-names label-names contexts]
   (module sym {:data-names data-names :label-names label-names :contexts contexts}))
  ([sym]
   (module sym {})))

(defn data-names [mod]
  (.dataNames mod))

(defn data-shapes [mod]
  (.dataShapes mod))

(defn label-shapes [mod]
  (.labelShapes mod))

(defn output-names [mod]
  (.outputNames mod))

(defn output-shapes [mod]
  (.outputShapes mod))

(s/def ::data-shapes (s/coll-of ::mx-io/data-desc))
(s/def ::label-shapes (s/coll-of ::mx-io/data-desc))
(s/def ::for-training boolean?)
(s/def ::inputs-need-grad boolean?)
(s/def ::force-rebind boolean?)
(s/def ::shared-module #(instance? Module))
(s/def ::grad-req string?)
(s/def ::bind-opts
  (s/keys :req-un [::data-shapes]
          :opt-un [::label-shapes ::for-training ::inputs-need-grad
                   ::force-rebind ::shared-module ::grad-req]))

(defn bind
  "Bind the symbols to construct executors. This is necessary before one
   can perform computation with the module.
    `mod`: module
    `opts-map` {
      `data-shapes`: map of `:name`, `:shape`, `:dtype`, and `:layout`
          Typically is `(provide-data-desc data-iter)`.Data shape must be in the
          form of `io/data-desc`
      `label-shapes`: map of `:name` `:shape` `:dtype` and `:layout`
          Typically is `(provide-label-desc data-iter)`.
      `for-training`: boolean - Default is `true`
          Whether the executors should be bind for training.
      `inputs-need-grad`: boolean - Default is `false`.
          Whether the gradients to the input data need to be computed.
          Typically this is not needed. But this might be needed when
          implementing composition of modules.
      `force-rebind`: boolean - Default is `false`.
          This function does nothing if the executors are already binded. But
          with this `true`, the executors will be forced to rebind.
      `shared-module`: Default is nil.
          This is used in bucketing. When not `nil`, the shared module
          essentially corresponds to a different bucket -- a module with
          different symbol but with the same sets of parameters (e.g. unrolled
          RNNs with different lengths).
    }
   Ex:
     (bind {:data-shapes (mx-io/provide-data train-iter)
            :label-shapes (mx-io/provide-label test-iter)})) "
  [mod {:keys [data-shapes label-shapes for-training inputs-need-grad
               force-rebind shared-module grad-req] :as opts
        :or {for-training true
             inputs-need-grad false
             force-rebind false
             grad-req "write"}}]
  (util/validate! ::bind-opts opts "Incorrect bind options")
  (doto mod
    (.bind
     (->> data-shapes
          (map mx-io/data-desc)
          (util/vec->indexed-seq))
     (util/->option (some->> label-shapes
                             (map mx-io/data-desc)
                             (util/vec->indexed-seq)))
     for-training
     inputs-need-grad
     force-rebind
     (util/->option shared-module)
     grad-req)))

(s/def ::intializer #(instance? Initializer %))
(s/def ::arg-params map?)
(s/def ::aux-params map?)
(s/def ::force-init boolean?)
(s/def ::allow-extra boolean?)
(s/def ::init-params-opts
  (s/keys :opt-un [::initializer ::arg-params ::aux-params
                   ::force-init ::allow-extra]))

(defn init-params
  "Initialize the parameters and auxiliary states.
    `opts-map` {
      `initializer`: Initializer - Default is `uniform`
          Called to initialize parameters if needed.
      `arg-params`: map
          If not nil, should be a map of existing arg-params. Initialization
          will be copied from that.
      `aux-params`: map
          If not nil, should be a map of existing aux-params. Initialization
          will be copied from that.
      `allow-missing`: boolean - Default is `false`
          If true, params could contain missing values, and the initializer will
          be called to fill those missing params.
      `force-init` boolean - Default is `false`
          If true, will force re-initialize even if already initialized.
      `allow-extra`: boolean - Default is `false`
          Whether allow extra parameters that are not needed by symbol.
          If this is `true`, no error will be thrown when `arg-params` or
          `aux-params` contain extra parameters that is not needed by the
          executor.
   Ex:
     (init-params {:initializer (initializer/xavier)})
     (init-params {:force-init true :allow-extra true})"
  ([mod {:keys [initializer arg-params aux-params allow-missing force-init
                allow-extra] :as opts
         :or {initializer (initializer/uniform 0.01)
              allow-missing false
              force-init false
              allow-extra false}}]
   (util/validate! ::init-params-opts opts "Invalid init-params opts")
   (doto mod
     (.initParams
      initializer
      (some-> arg-params (util/convert-map))
      (some-> aux-params (util/convert-map))
      allow-missing
      force-init
      allow-extra)))
  ([mod]
   (init-params mod {})))

(s/def ::optimizer #(instance? Optimizer %))
(s/def ::kvstore string?)
(s/def ::reset-optimizer boolean?)
(s/def ::force-init boolean?)
(s/def ::init-optimizer-opts
  (s/keys :opt-un [::optimizer ::kvstore ::reset-optimizer ::force-init]))

(defn init-optimizer
  "Install and initialize optimizers.
    `mod`: Module
    `opts-map` {
      `kvstore`: string - Default is \"local\"
      `optimizer`: Optimizer - Default is `sgd`
      `reset-optimizer`: boolean - Default is `true`
          Indicating whether we should set `rescaleGrad` & `idx2name` for
          optimizer according to executorGroup.
      `force-init`: boolean - Default is `false`
          Indicating whether we should force re-initializing the optimizer
          in the case an optimizer is already installed.
   Ex:
     (init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})"
  ([mod {:keys [kvstore optimizer reset-optimizer force-init] :as opts
         :or {kvstore "local"
              optimizer (optimizer/sgd)
              reset-optimizer true
              force-init false}}]
   (util/validate! ::init-optimizer-opts opts "Invalid init-optimizer options")
   (doto mod
     (.initOptimizer kvstore optimizer reset-optimizer force-init)))
  ([mod]
   (init-optimizer mod {})))

(defn forward
  "Forward computation.
    `data-batch`: Either map or DataBatch
       Input data of form `io/data-batch`.
    `is-train`: Default is nil
       Which means `is_train` takes the value of `for_training`."
  ([mod data-batch is-train]
   (util/validate! ::mx-io/data-batch data-batch "Invalid data batch")
   (doto mod
     (.forward
      (if (map? data-batch)
        (mx-io/data-batch data-batch)
        data-batch)
      (util/->option is-train))))
  ([mod data-batch-map]
   (forward mod data-batch-map nil)))

(s/def ::ndarray #(instance? NDArray %))
(s/def ::out-grads (s/nilable (s/coll-of ::ndarray)))

(defn backward
  "Backward computation.
    `out-grads`: collection of NDArrays
        Gradient on the outputs to be propagated back. This parameter is only
        needed when bind is called on outputs that are not a loss function."
  ([mod out-grads]
   (util/validate! ::out-grads out-grads "Invalid out-grads")
   (doto mod
     (.backward (some-> out-grads into-array))))
  ([mod]
   (backward mod nil)))

(defn forward-backward
  "A convenient function that calls both `forward` and `backward`."
  [mod data-batch]
  (util/validate! ::mx-io/data-batch data-batch "Invalid data-batch")
  (doto mod
    (.forwardBackward data-batch)))

(defn outputs
  "Get outputs of the previous forward computation.
   In the case when data-parallelism is used, the outputs will be collected from
   multiple devices. The results will look like
   `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`.
   Those `NDArray`s might live on different devices."
  [mod]
  (->> (.getOutputs mod)
       (util/scala-vector->vec)
       (mapv util/scala-vector->vec)))

(defn update
  "Update parameters according to the installed optimizer and the gradients
   computed in the previous forward-backward batch."
  [mod]
  (doto mod
    (.update)))

(defn outputs-merged
  "Get outputs of the previous forward computation.
   In the case when data-parallelism is used, the outputs will be merged from
   multiple devices, as they look like from a single executor.
   The results will look like `[out1, out2]`."
  [mod]
  (->> (.getOutputsMerged mod)
       (util/scala-vector->vec)))

(defn input-grads
  "Get the gradients to the inputs, computed in the previous backward computation.
   In the case when data-parallelism is used, the outputs will be collected from
   multiple devices. The results will look like
   `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`.
   Those `NDArray`s might live on different devices."
  [mod]
  (->> (.getInputGrads mod)
       (util/scala-vector->vec)
       (mapv util/scala-vector->vec)))

(defn input-grads-merged
  "Get the gradients to the inputs, computed in the previous backward computation.
   In the case when data-parallelism is used, the outputs will be merged from
   multiple devices, as they look like from a single executor.
   The results will look like `[grad1, grad2]`."
  [mod]
  (->> (.getInputGradsMerged mod)
       (util/scala-vector->vec)))

(s/def ::prefix string?)
(s/def ::epoch int?)
(s/def ::save-opt-states boolean?)
(s/def ::save-checkpoint-opts
  (s/keys :req-un [::prefix ::epoch]
          :opt-un [::save-opt-states ::save-checkpoint]))

(defn save-checkpoint
  "Save current progress to checkpoint.
   Use mx.callback.module_checkpoint as epoch_end_callback to save during
   training.
    `mod`: Module
    `opts-map` {
       `prefix`: string
           The file prefix to checkpoint to
       `epoch`: int
           The current epoch number
       `save-opt-states`: boolean - Default is `false`
           Whether to save optimizer states for continue training
    }
   Ex:
     (save-checkpoint {:prefix \"saved_model\" :epoch 0 :save-opt-states true})"
  ([mod {:keys [prefix epoch save-opt-states] :as opts
         :or {save-opt-states false}}]
   (util/validate! ::save-checkpoint-opts opts "Invalid save checkpoint opts")
   (doto mod
     (.saveCheckpoint prefix (int epoch) save-opt-states)))
  ([mod prefix epoch]
   (save-checkpoint mod {:prefix prefix :epoch epoch})))

(s/def ::load-optimizer-states boolean?)
(s/def ::data-names (s/coll-of string? :kind vector?))
(s/def ::label-names (s/coll-of string? :kind vector?))
(s/def ::context #(instance? Context %))
(s/def ::contexts (s/coll-of ::context :kind vector?))
(s/def ::workload-list (s/coll-of number? :kind vector?))
(s/def ::fixed-params-names (s/coll-of string? :kind vector?))
(s/def ::load-checkpoint-opts
  (s/keys :req-un [::prefix ::epoch]
          :opt-un [::load-optimizer-states ::data-names ::label-names
                   ::contexts ::workload-list ::fixed-param-names]))

(defn load-checkpoint
  "Create a model from previously saved checkpoint.
    `opts-map` {
      `prefix`: string
          Path prefix of saved model files. You should have prefix-symbol.json,
          prefix-xxxx.params, and optionally prefix-xxxx.states, where xxxx is
          the epoch number.
      `epoch`: int
          Epoch to load.
      `load-optimizer-states`: boolean - Default is false
           Whether to load optimizer states. Checkpoint needs to have been made
           with `save-optimizer-states` = `true`.
       `data-names`: vector of strings - Default is [\"data\"]
           Input data names.
       `label-names`: vector of strings - Default is [\"softmax_label\"]
           Input label names.
       `contexts`: Context - Default is `context/cpu`
       `workload-list`:  Default nil
           Indicating uniform workload.
       `fixed-param-names`: Default nil
           Indicating no network parameters are fixed.
   Ex:
     (load-checkpoint {:prefix \"my-model\" :epoch 1 :load-optimizer-states true}"
  ([{:keys [prefix epoch load-optimizer-states data-names label-names contexts
            workload-list fixed-param-names] :as opts
     :or {load-optimizer-states false
          data-names ["data"]
          label-names ["softmax_label"]
          contexts [(context/cpu)]
          workload-list nil
          fixed-param-names nil}}]
   (util/validate! ::load-checkpoint-opts opts "Invalid load-checkpoint opts")
   (Module/loadCheckpoint
    prefix
    (int epoch)
    load-optimizer-states
    (util/vec->indexed-seq data-names)
    (util/vec->indexed-seq label-names)
    (into-array contexts)
    (util/->option (when workload-list (util/vec->indexed-seq workload-list)))
    (util/->option (when fixed-param-names (util/vec->set fixed-param-names)))))
  ([prefix epoch]
   (load-checkpoint {:prefix prefix :epoch epoch})))

(defn load-optimizer-states [mod fname]
  (.mod load fname))

(defn symbol [mod]
  (.getSymbol mod))

(defn params [mod]
  (map util/scala-map->map (util/coerce-return (.getParams mod))))

(defn arg-params [mod]
  (util/scala-map->map (.argParams mod)))

(defn aux-params [mod]
  (util/scala-map->map (.auxParams mod)))

(defn reshape
  "Reshapes the module for new input shapes.
    `mod`: Module
    `data-shapes`: Typically is `(provide-data data-iter)`
    `label-shapes`: Typically is `(provide-label data-tier)`"
  ([mod data-shapes label-shapes]
   (util/validate! ::data-shapes data-shapes "Invalid data-shapes")
   (util/validate! (s/nilable ::label-shapes) label-shapes "Invalid label-shapes")
   (doto mod
     (.reshape
      (->> data-shapes
           (map mx-io/data-desc)
           (util/vec->indexed-seq))
      (util/->option (some->> label-shapes
                              (map mx-shape/->shape)
                              (util/vec->indexed-seq))))))
  ([mod data-shapes]
   (reshape mod data-shapes nil)))

(s/def ::set-param-opts
  (s/keys :opt-un [::arg-params ::aux-params ::allow-missing
                   ::force-init ::allow-extra]))

(defn get-params [mod]
  (.getParams mod))

(defn set-params
  "Assign parameters and aux state values.
    `mod`: Module
    `opts-map` {
      `arg-params`: map - map of name to value (`NDArray`) mapping.
      `aux-params`: map - map of name to value (`NDArray`) mapping.
      `allow-missing`: boolean
          If true, params could contain missing values, and the initializer will
          be called to fill those missing params.
      `force-init`: boolean - Default is `false`
          If true, will force re-initialize even if already initialized.
      `allow-extra`: boolean - Default is `false`
          Whether allow extra parameters that are not needed by symbol. If this
          is `true`, no error will be thrown when arg-params or aux-params
          contain extra parameters that is not needed by the executor.
    }
   Ex:
     (set-params mod
       {:arg-params {\"fc_0_weight\" (ndarray/array [0.15 0.2 0.25 0.3] [2 2])
        :allow-missing true})"
  [mod {:keys [arg-params aux-params allow-missing force-init
               allow-extra] :as opts
        :or {allow-missing false force-init true allow-extra false}}]
  (util/validate! ::set-param-opts opts "Invalid set-params")
  (doto mod
    (.setParams
     (util/convert-symbol-map arg-params)
     (when aux-params (util/convert-symbol-map aux-params))
     allow-missing
     force-init
     allow-extra)))

(defn install-monitor
  "Install monitor on all executors."
  [mod monitor]
  (doto mod
    (.installMonitor monitor)))

(defn borrow-optimizer
  "Borrow optimizer from a shared module. Used in bucketing, where exactly the
   same optimizer (esp. kvstore) is used.
    `mod`: Module
    `shared-module`"
  [mod shared-module]
  (doto mod
    (.borrowOptimizer shared-module)))

(defn save-optimizer-states
  "Save optimizer (updater) state to file.
    `mod`: Module
    `fname`: string - Path to output states file."
  [mod fname]
  (doto mod
    (.saveOptimizerStates mod fname)))

(defn load-optimizer-states
  "Load optimizer (updater) state from file.
    `mod`: Module
    `fname`: string - Path to input states file."
  [mod fname]
  (doto mod
    (.loadOptimzerStates fname)))

(s/def ::eval-metric #(instance? EvalMetric %))
(s/def ::labels (s/coll-of ::ndarray :kind vector?))

(defn update-metric
  "Evaluate and accumulate evaluation metric on outputs of the last forward
   computation.
     `mod`: module
     `eval-metric`: EvalMetric
     `labels`: collection of NDArrays
  Ex:
    (update-metric mod (eval-metric/mse) labels)"
  [mod eval-metric labels]
  (util/validate! ::eval-metric eval-metric "Invalid eval metric")
  (util/validate! ::labels labels "Invalid labels")
  (doto mod
    (.updateMetric eval-metric (util/vec->indexed-seq labels))))

(s/def ::begin-epoch int?)
(s/def ::validation-metric ::eval-metric)
(s/def ::monitor #(instance? Monitor %))
(s/def ::batch-end-callback #(instance? Callback$Speedometer %))
(s/def ::fit-params-opts
  (s/keys :opt-un [::eval-metric ::kvstore ::optimizer ::initializer
                   ::arg-params ::aux-params ::allow-missing ::force-rebind
                   ::force-init ::begin-epoch ::validation-metric ::monitor
                   ::batch-end-callback]))

;; callbacks are not supported for now
(defn fit-params
  "Initialize FitParams with provided parameters.
    `eval-metric`: EvalMetric - Default is `accuracy`
    `kvstore`: String - Default is \"local\"
    `optimizer`: Optimizer - Default is `sgd`
    `initializer`: Initializer - Default is `uniform`
        Called to initialize parameters if needed.
    `arg-params`: map
        If not nil, should be a map of existing `arg-params`. Initialization
        will be copied from that.
    `aux-params`: map -
        If not nil, should be a map of existing `aux-params`. Initialization
        will be copied from that.
    `allow-missing`: boolean - Default is `false`
        If `true`, params could contain missing values, and the initializer will
        be called to fill those missing params.
    `force-rebind`: boolean - Default is `false`
        This function does nothing if the executors are already binded. But with
        this `true`, the executors will be forced to rebind.
    `force-init`: boolean - Default is `false`
        If `true`, will force re-initialize even if already initialized.
    `begin-epoch`: int - Default is 0
    `validation-metric`: EvalMetric
    `monitor`: Monitor
  Ex:
    (fit-params {:force-init true :force-rebind true :allow-missing true})
    (fit-params
      {:batch-end-callback (callback/speedometer batch-size 100)
       :initializer (initializer/xavier)
       :optimizer (optimizer/sgd {:learning-rate 0.01})
       :eval-metric (eval-metric/mse)})"
  ([{:keys [eval-metric kvstore optimizer
            initializer arg-params aux-params
            allow-missing force-rebind force-init begin-epoch
            validation-metric monitor batch-end-callback] :as opts
     :or {eval-metric (eval-metric/accuracy)
          kvstore "local"
          optimizer (optimizer/sgd)
          initializer (initializer/uniform 0.01)
          allow-missing false
          force-rebind false
          force-init false
          begin-epoch 0}}]
   (util/validate! ::fit-params-opts opts "Invalid fit param opts")
   (doto (new FitParams)
     (.setEvalMetric eval-metric)
     (.setKVStore kvstore)
     (.setOptimizer optimizer)
     (.setInitializer initializer)
     (.setArgParams (some-> arg-params (util/convert-map)))
     (.setAuxParams (some-> aux-params (util/convert-map)))
     (.setAllowMissing allow-missing)
     (.setForceRebind force-rebind)
     (.setForceInit force-init)
     (.setBeginEpoch (int begin-epoch))
     (.setValidationMetric validation-metric)
     (.setMonitor monitor)
     (.setBatchEndCallback batch-end-callback)))
  ([]
   (new FitParams)))

(s/def ::mx-data-iter #(instance? MXDataIter %))
(s/def ::ndarray-iter #(instance? NDArrayIter %))
(s/def ::train-data (s/or :mx-iter ::mx-data-iter :ndarry-iter ::ndarray-iter))
(s/def ::eval-data ::train-data)
(s/def ::num-epoch (s/and int? pos?))
(s/def ::fit-params #(instance? FitParams %))
(s/def ::fit-options
  (s/keys :req-un [::train-data]
          :opt-un [::eval-data ::num-epoch ::fit-params]))

;;; High Level API

(defn score
  "Run prediction on `eval-data` and evaluate the performance according to
  `eval-metric`.
    `mod`: module
     `opts-map` {
       `eval-data`: DataIter
       `eval-metric`: EvalMetric
       `num-batch`: int - Default is `Integer.MAX_VALUE`
           Number of batches to run. Indicating run until the `DataIter`
           finishes.
       `batch-end-callback`: not supported yet.
       `reset`: boolean - Default is `true`,
           Indicating whether we should reset `eval-data` before starting
           evaluating.
       `epoch`: int - Default is 0
           For compatibility, this will be passed to callbacks (if any). During
           training, this will correspond to the training epoch number.
     }
   Ex:
     (score mod {:eval-data data-iter :eval-metric (eval-metric/accuracy)})
     (score mod {:eval-data data-iter
                 :eval-metric (eval-metric/mse) :num-batch 10})"
  [mod {:keys [eval-data eval-metric num-batch reset epoch] :as opts
        :or {num-batch Integer/MAX_VALUE
             reset true
             epoch 0}}]
  (util/validate! ::score-opts opts "Invalid score options")
  (do (eval-metric/reset eval-metric)
      (eval-metric/get
       (.score mod
               eval-data
               eval-metric
               (int num-batch)
               (util/->option nil)
               (util/->option nil)
               reset
               (int epoch)))))

(defn fit
  "Train the module parameters.
    `mod`: Module
    `opts-map` {
      `train-data`: DataIter
      `eval-data`: DataIter
          If not nil, will be used as validation set and evaluate the
          performance after each epoch.
      `num-epoch`: int
          Number of epochs to run training.
      `fit-params`: FitParams
          Extra parameters for training (see fit-params).
    }
   Ex:
     (fit {:train-data train-iter :eval-data test-iter :num-epoch 100)
     (fit {:train-data train-iter
           :eval-data test-iter
           :num-epoch 5
           :fit-params
           (fit-params {:batch-end-callback (callback/speedometer 128 100)
                        :initializer (initializer/xavier)
                        :optimizer (optimizer/sgd {:learning-rate 0.01})
                        :eval-metric (eval-metric/mse)}))"
  [mod {:keys [train-data eval-data num-epoch fit-params] :as opts
        :or {num-epoch 1
             fit-params (new FitParams)}}]
  (util/validate! ::fit-options opts "Invalid options for fit")
  (doto mod
    (.fit
     train-data
     (util/->option eval-data)
     (int num-epoch)
     fit-params)))

(s/def ::eval-data ::train-data)
(s/def ::num-batch integer?)
(s/def ::reset boolean?)
(s/def ::predict-opts
  (s/keys :req-un [::eval-data] :opt-un [::num-batch ::reset]))

(defn predict-batch
  "Run the predication on a data batch.
    `mod`: Module
    `data-batch`: data-batch"
  [mod data-batch]
  (util/validate! ::mx-io/data-batch data-batch "Invalid data batch")
  (util/coerce-return (.predict mod (if (map? data-batch)
                                      (mx-io/data-batch data-batch)
                                      data-batch))))

(defn predict
  "Run prediction and collect the outputs.
    `mod`: Module
    `opts-map` {
      `eval-data`: DataIter
      `num-batch` int - Default is `-1`
          Indicating running all the batches in the data iterator.
      `reset`: boolean - Default is `true`
          Indicating whether we should reset the data iter before start doing
          prediction.
    }
    returns: vector of NDArrays `[out1, out2, out3]` where each element is the
       concatenation of the outputs for all the mini-batches.
  Ex:
    (predict mod {:eval-data test-iter})
    (predict mod {:eval-data test-iter :num-batch 10 :reset false})"
  [mod {:keys [eval-data num-batch reset] :as opts
        :or {num-batch -1
             reset true}}]
  (util/validate! ::predict-opts opts "Invalid opts for predict")
  (util/scala-vector->vec (.predict mod eval-data (int num-batch) reset)))

(s/def ::predict-every-batch-opts
  (s/keys :req-un [::eval-data] :opt-un [::num-batch ::reset]))

(defn predict-every-batch
  "Run prediction and collect the outputs.
    `mod`: Module
    `opts-map` {
      `eval-data`: DataIter
      `num-batch` int - Default is `-1`
          Indicating running all the batches in the data iterator.
      `reset` boolean - Default is `true`
          Indicating whether we should reset the data iter before start doing
          prediction.
    }
    returns: nested list like this
    `[[out1_batch1, out2_batch1, ...], [out1_batch2, out2_batch2, ...]]`

   Note: This mode is useful because in some cases (e.g. bucketing), the module
         does not necessarily produce the same number of outputs.
   Ex:
     (predict-every-batch mod {:eval-data test-iter})"
  [mod {:keys [eval-data num-batch reset] :as opts
        :or {num-batch -1
             reset true}}]
  (util/validate! ::predict-every-batch-opts
                  opts
                  "Invalid opts for predict-every-batch")
  (mapv util/scala-vector->vec
        (util/scala-vector->vec
          (.predictEveryBatch mod eval-data (int num-batch) reset))))

(s/def ::score-opts
  (s/keys :req-un [::eval-data ::eval-metric]
          :opt-un [::num-batch ::reset ::epoch]))

(defn exec-group [mod]
  (.execGroup mod))

(defn grad-arrays [mod]
  (mapv vec (util/buffer->vec (.gradArrays (.execGroup mod)))))

(comment
  (require '[clojure.reflect :as r])
  (r/reflect DataDesc)
  (new DataDesc)

  (.setEpochEndCallback (if epoch-end-callback
                          (util/->option epoch-end-callback)
                          (util/->option nil)))
  (.setBatchEndCallback (if batch-end-callback
                          (util/->option batch-end-callback)
                          (util/->option nil)))

  (fit-params {:allow-missing true})
  (fit-params {}))
