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
                             Context EvalMetric Monitor Callback$Speedometer DataDesc)))

(defn module
  "Module is a basic module that wrap a symbol.
   sym : Symbol definition.
   map of options
       :data-names - Input data names.
       :label-names - Input label names
       :contexts - Default is cpu().
       :workload-list - Default nil, indicating uniform workload.
       :fixed-param-names Default nil, indicating no network parameters are fixed."
  ([sym {:keys [data-names label-names contexts workload-list fixed-param-names] :as opts
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
(s/def ::bind-opts (s/keys :req-un [::data-shapes] :opt-un [::label-shapes ::for-training ::inputs-need-grad
                                                            ::force-rebind ::shared-module ::grad-req]))

(defn bind
  "Bind the symbols to construct executors. This is necessary before one
   can perform computation with the module.
   mod : module
   map of opts:
     :data-shapes Typically is  (provide-data data-iter). Data shape must be in the form of io/data-desc with is a map of :name :shape :dtype and :layout
     :label-shapes Typically is  (provide-label data-iter). map of :name :shape :dtype and :layout
     :for-training Default is `true`. Whether the executors should be bind for training.
     :inputs-need-grad Default is `false`.
                       Whether the gradients to the input data need to be computed.
                       Typically this is not needed.
                       But this might be needed when implementing composition of modules.
     :force-rebind Default is `false`.
                   This function does nothing if the executors are already binded.
                   But with this `true`, the executors will be forced to rebind.
     :shared-module Default is nil. This is used in bucketing.
                    When not `None`, the shared module essentially corresponds to
                    a different bucket -- a module with different symbol
                    but with the same sets of parameters
                    (e.g. unrolled RNNs with different lengths). "
  [mod {:keys [data-shapes label-shapes for-training inputs-need-grad force-rebind
               shared-module grad-req] :as opts
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
(s/def ::init-params-opts (s/keys :opt-un [::initializer ::arg-params ::aux-params
                                           ::force-init ::allow-extra]))

(defn init-params
  " Initialize the parameters and auxiliary states.
   options map
     :initializer - Called to initialize parameters if needed.
     :arg-params -  If not nil, should be a map of existing arg-params.
                     Initialization will be copied from that.
     :auxParams - If not nil, should be a map of existing aux-params.
                    Initialization will be copied from that.
     :allow-missing - If true, params could contain missing values,
                       and the initializer will be called to fill those missing params.
     :force-init -  If true, will force re-initialize even if already initialized.
     :allow-extra -  Whether allow extra parameters that are not needed by symbol.
             If this is True, no error will be thrown when argParams or auxParams
             contain extra parameters that is not needed by the executor."
  ([mod {:keys [initializer arg-params aux-params allow-missing force-init allow-extra] :as opts
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
(s/def ::init-optimizer-opts (s/keys :opt-un [::optimizer ::kvstore ::reset-optimizer ::force-init]))

(defn init-optimizer
  " Install and initialize optimizers.
   - mod Module
   - options map of
          - kvstore
         - reset-optimizer Default `True`, indicating whether we should set
           `rescaleGrad` & `idx2name` for optimizer according to executorGroup
         -  force-init Default `False`, indicating whether we should force
             re-initializing the optimizer in the case an optimizer is already installed."
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
   data-batch -  input data of form io/data-batch either map or DataBatch
   is-train -  Default is nil, which means `is_train` takes the value of `for_training`."
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
   out-grads -  Gradient on the outputs to be propagated back.
                This parameter is only needed when bind is called
                on outputs that are not a loss function."
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
  " Get outputs of the previous forward computation.
  In the case when data-parallelism is used,
            the outputs will be collected from multiple devices.
            The results will look like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`,
           those `NDArray` might live on different devices."
  [mod]
  (->> (.getOutputs mod)
       (util/scala-vector->vec)
       (mapv util/scala-vector->vec)))

(defn update
  "Update parameters according to the installed optimizer and the gradients computed
   in the previous forward-backward batch."
  [mod]
  (doto mod
    (.update)))

(defn outputs-merged
  " Get outputs of the previous forward computation.
    return In the case when data-parallelism is used,
            the outputs will be merged from multiple devices,
            as they look like from a single executor.
            The results will look like `[out1, out2]`"
  [mod]
  (->> (.getOutputsMerged mod)
       (util/scala-vector->vec)))

(defn input-grads
  "  Get the gradients to the inputs, computed in the previous backward computation.
  In the case when data-parallelism is used,
            the outputs will be collected from multiple devices.
            The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`
           those `NDArray` might live on different devices."
  [mod]
  (->> (.getInputGrads mod)
       (util/scala-vector->vec)
       (mapv util/scala-vector->vec)))

(defn input-grads-merged
  " Get the gradients to the inputs, computed in the previous backward computation.
    return In the case when data-parallelism is used,
            the outputs will be merged from multiple devices,
            as they look like from a single executor.
            The results will look like `[grad1, grad2]`"
  [mod]
  (->> (.getInputGradsMerged mod)
       (util/scala-vector->vec)))

(s/def ::prefix string?)
(s/def ::epoch int?)
(s/def ::save-opt-states boolean?)
(s/def ::save-checkpoint-opts (s/keys :req-un [::prefix ::epoch] :opt-un [::save-opt-states ::save-checkpoint]))

(defn save-checkpoint
  " Save current progress to checkpoint.
    Use mx.callback.module_checkpoint as epoch_end_callback to save during training.
    - mod Module
    -  opt-map with
       :prefix The file prefix to checkpoint to
       :epoch The current epoch number
       :save-opt-states Whether to save optimizer states for continue training "
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
(s/def ::load-checkpoint-opts (s/keys :req-un [::prefix ::epoch]
                                      :opt-un [::load-optimizer-states ::data-names ::label-names
                                               ::contexts ::workload-list ::fixed-param-names]))

(defn load-checkpoint
  "Create a model from previously saved checkpoint.
   - opts map of
     -  prefix Path prefix of saved model files. You should have prefix-symbol.json,
                 prefix-xxxx.params, and optionally prefix-xxxx.states,
                 where xxxx is the epoch number.
     -  epoch Epoch to load.
     - load-optimizer-states Whether to load optimizer states.
                          Checkpoint needs to have been made with save-optimizer-states=True
     - dataNames Input data names.
     - labelNames Input label names
     - contexts Default is cpu().
     -  workload-list  Default nil, indicating uniform workload.
     - fixed-param-names Default nil, indicating no network parameters are fixed."
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
  " Reshapes the module for new input shapes.
   - mod module
   - data-shapes Typically is `(provide-data data-iter)
   - param label-shapes Typically is `(provide-label data-tier)`. "
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

(s/def ::set-param-opts (s/keys :opt-un [::arg-params ::aux-params ::allow-missing ::force-init ::allow-extra]))

(defn get-params [mod]
  (.getParams mod))

(defn set-params
  " Assign parameter and aux state values.
    - mod module
    - arg-params : map
            map of name to value (`NDArray`) mapping.
    - aux-params : map
           map of name to value (`NDArray`) mapping.
    - allow-missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
    - force-init : bool
            If true, will force re-initialize even if already initialized.
   -  allow-extra : bool
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg-params or aux-params
            contain extra parameters that is not needed by the executor."
  [mod {:keys [arg-params aux-params allow-missing force-init allow-extra] :as opts
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
  "Install monitor on all executors"
  [mod monitor]
  (doto mod
    (.installMonitor monitor)))

(defn borrow-optimizer
  "Borrow optimizer from a shared module. Used in bucketing, where exactly the same
   optimizer (esp. kvstore) is used.
   - mod module
   - shared-module"
  [mod shared-module]
  (doto mod
    (.borrowOptimizer shared-module)))

(defn save-optimizer-states
  "Save optimizer (updater) state to file
   - mod module
   - fname Path to output states file."
  [mod fname]
  (doto mod
    (.saveOptimizerStates mod fname)))

(defn load-optimizer-states
  "Load optimizer (updater) state from file
   - mod module
   - fname Path to input states file.
  "
  [mod fname]
  (doto mod
    (.loadOptimzerStates fname)))

(s/def ::eval-metric #(instance? EvalMetric %))
(s/def ::labels (s/coll-of ::ndarray :kind vector?))

(defn update-metric
  "Evaluate and accumulate evaluation metric on outputs of the last forward computation.
    - mod module
    - eval-metric
    - labels"
  [mod eval-metric labels]
  (util/validate! ::eval-metric eval-metric "Invalid eval metric")
  (util/validate! ::labels labels "Invalid labels")
  (doto mod
    (.updateMetric eval-metric (util/vec->indexed-seq labels))))

(s/def ::begin-epoch int?)
(s/def ::validation-metric ::eval-metric)
(s/def ::monitor #(instance? Monitor %))
(s/def ::batch-end-callback #(instance? Callback$Speedometer %))
(s/def ::fit-params-opts (s/keys :opt-un [::eval-metric ::kvstore ::optimizer ::initializer
                                          ::arg-params ::aux-params ::allow-missing ::force-rebind
                                          ::force-init ::begin-epoch ::validation-metric ::monitor
                                          ::batch-end-callback]))

;; callbacks are not supported for now
(defn fit-params
  "Fit Params"
  ([{:keys [eval-metric kvstore optimizer
            initializer arg-params aux-params
            allow-missing force-rebind force-init begin-epoch validation-metric monitor
            batch-end-callback] :as opts
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
(s/def ::num-epoch int?)
(s/def ::fit-params #(instance? FitParams %))
(s/def ::fit-options (s/keys :req-un [::train-data] :opt-un [::eval-data ::num-epoch ::fit-params]))

;;; High Level API

(defn score
  " Run prediction on `eval-data` and evaluate the performance according to `eval-metric`.
   - mod module
   - option map with
     :eval-data : DataIter
     :eval-metric : EvalMetric
     :num-batch Number of batches to run. Default is `Integer.MAX_VALUE`,
                   indicating run until the `DataIter` finishes.
     :batch-end-callback -not supported yet
     :reset Default `True`,
                 indicating whether we should reset `eval-data` before starting evaluating.
     :epoch Default 0. For compatibility, this will be passed to callbacks (if any).
                During training, this will correspond to the training epoch number."
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
   - mod module
   - train-data (data-iterator)
   - eval-data (data-iterator)If not nil, will be used as validation set and evaluate
                   the performance after each epoch.
   - num-epoch Number of epochs to run training.
   - f-params Extra parameters for training (See fit-params)."
  [mod {:keys [train-data eval-data num-epoch fit-params] :as opts
        `:or {num-epoch 1
              fit-params (new FitParams)}}]
  (util/validate! ::fit-options opts "Invalid options for fit")
  (let [fmod (-> mod
                 (bind {:data-shapes (mx-io/provide-data train-data)
                        :label-shapes (mx-io/provide-label train-data)
                        :for-training true
                        :force-rebind (.forceRebind fit-params)})
                 (init-params (remove (fn [[k v]] (nil? v))
                                      {:initializer (.initializer fit-params)
                                       :arg-params (.argParams fit-params)
                                       :aux-params (.auxParams fit-params)
                                       :allow-missing (.allowMissing fit-params)}))
                 (init-optimizer (remove (fn [[k v]] (nil? v))
                                         {:optimizer (.optimizer fit-params)
                                          :kvstore (.kvstore fit-params)})))
        eval-metric (or (.evalMetric fit-params) (eval-metric/accuracy))
        val-metric (or (util/option->value (.validationMetric fit-params)) (eval-metric/accuracy))]
    (doseq [i (range num-epoch)]
      (let [tic (System/currentTimeMillis)]
        (mx-io/reduce-batches train-data
                              (fn [batch-num batch]
                                (-> fmod
                                    (forward batch)
                                    (backward)
                                    (update)
                                    (update-metric eval-metric (mx-io/batch-label batch)))
                                (when-let [cb (util/option->value (.batchEndCallback fit-params))]
                                  (callback/invoke cb i batch-num eval-metric))
                                (.dispose batch)
                                (inc batch-num)))
        (println "Epoch " i " Train-" (eval-metric/get eval-metric))
        (println "Epoch " i " Time cost-" (- (System/currentTimeMillis) tic))

       ;;sync across kvstores
        (get-params fmod)
        (when-let [cb (util/option->value (.epochEndCallback fit-params))]
          (callback/invoke cb i 0 val-metric))

       ;; evaluation on the validation set
        (when eval-data
          (let [res (score fmod {:eval-data eval-data :eval-metric eval-metric :epoch i})]
            (println "Epoch " i " Validation- " res)))))
    fmod)
  ;; old way if the problem with the sizes get resolved in DataDesc
  #_(doto mod
      (.fit
       train-data
       (util/->option eval-data)
       (int num-epoch)
       fit-params)))

(s/def ::eval-data ::train-data)
(s/def ::num-batch integer?)
(s/def ::reset boolean?)
(s/def ::predict-opts (s/keys :req-un [::eval-data] :opt-un [::num-batch ::reset]))

(defn predict-batch
  "Run the predication on a data batch
   - mod module
   - data-batch data-batch"
  [mod data-batch]
  (util/validate! ::mx-io/data-batch data-batch "Invalid data batch")
  (util/coerce-return (.predict mod (if (map? data-batch)
                                      (mx-io/data-batch data-batch)
                                      data-batch))))

(defn predict
  "Run prediction and collect the outputs.
   - mod module
   - option map with
     - :eval-data
     - :num-batch Default is -1, indicating running all the batches in the data iterator.
     - :reset Default is `True`, indicating whether we should reset the data iter before start
               doing prediction.
    The return value will be a vector of NDArrays `[out1, out2, out3]`.
          Where each element is concatenation of the outputs for all the mini-batches."
  [mod {:keys [eval-data num-batch reset] :as opts
        :or {num-batch -1
             reset true}}]
  (util/validate! ::predict-opts opts "Invalid opts for predict")
  (util/scala-vector->vec (.predict mod eval-data (int num-batch) reset)))

(s/def ::predict-every-batch-opts (s/keys :req-un [::eval-data] :opt-un [::num-batch ::reset]))

(defn predict-every-batch
  " Run prediction and collect the outputs.
   - module
   - option map with
     :eval-data
     :num-batch Default is -1, indicating running all the batches in the data iterator.
     :reset Default is `True`, indicating whether we should reset the data iter before start
               doing prediction.
    The return value will be a nested list like
   [[out1_batch1, out2_batch1, ...], [out1_batch2, out2_batch2, ...]]`
   This mode is useful because in some cases (e.g. bucketing),
    the module does not necessarily produce the same number of outputs."
  [mod {:keys [eval-data num-batch reset] :as opts
        :or {num-batch -1
             reset true}}]
  (util/validate! ::predict-every-batch-opts opts "Invalid opts for predict-every-batch")
  (mapv util/scala-vector->vec (util/scala-vector->vec (.predictEveryBatch mod eval-data (int num-batch) reset))))

(s/def ::score-opts (s/keys :req-un [::eval-data ::eval-metric] :opt-un [::num-batch ::reset ::epoch]))

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
