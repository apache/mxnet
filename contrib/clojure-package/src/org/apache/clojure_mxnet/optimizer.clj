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

(ns org.apache.clojure-mxnet.optimizer
  (:refer-clojure :exclude [update])
  (:require  
   [clojure.spec.alpha :as s]
   [org.apache.clojure-mxnet.util :as util])
  (:import 
   (org.apache.mxnet.optimizer SGD DCASGD NAG AdaDelta RMSProp AdaGrad Adam SGLD)
   (org.apache.mxnet FactorScheduler)))

(s/def ::learning-rate number?)
(s/def ::momentum number?)
(s/def ::wd number?)
(s/def ::clip-gradient number?)
(s/def ::lr-scheduler #(instance? FactorScheduler %))
(s/def ::sgd-opts (s/keys :opt-un [::learning-rate ::momentum ::wd ::clip-gradient ::lr-scheduler]))

(defn sgd
  "A very simple SGD optimizer with momentum and weight regularization."
  ([{:keys [learning-rate momentum wd clip-gradient lr-scheduler] :as opts
     :or {learning-rate 0.01
          momentum 0.0
          wd 0.0001
          clip-gradient 0}}]
   (util/validate! ::sgd-opts opts "Incorrect sgd optimizer options")
   (new SGD (float learning-rate) (float momentum) (float wd) (float clip-gradient) lr-scheduler))
  ([]
   (sgd {})))

(s/def ::lambda number?)
(s/def ::dcasgd-opts (s/keys :opt-un [::learning-rate ::momentum ::lambda ::wd ::clip-gradient ::lr-scheduler]))

(defn dcasgd
  "DCASGD optimizer with momentum and weight regularization.
  Implementation of paper 'Asynchronous Stochastic Gradient Descent with
  Delay Compensation for Distributed Deep Learning'"
  ([{:keys [learning-rate momentum lambda wd clip-gradient lr-scheduler] :as opts
     :or {learning-rate 0.01
          momentum 0.0
          lambda 0.04
          wd 0.0
          clip-gradient 0}}]
   (util/validate! ::sgd-opts opts "Incorrect dcasgd optimizer options")
   (new DCASGD (float learning-rate) (float lambda) (float momentum) (float wd) (float clip-gradient) lr-scheduler))
  ([]
   (dcasgd {})))

(s/def ::nag-opts (s/keys :opt-un [::learning-rate ::momentum ::wd ::clip-gradient ::lr-scheduler]))

(defn nag
  "SGD with nesterov.
   It is implemented according to
   https://github.com/torch/optim/blob/master/sgd.lua"
  ([{:keys [learning-rate momentum wd clip-gradient lr-scheduler] :as opts
     :or {learning-rate 0.01
          momentum 0.0
          wd 0.0001
          clip-gradient 0}}]
   (util/validate! ::nag-opts opts "Incorrect nag optimizer options")
   (new NAG (float learning-rate) (float momentum) (float wd) (float clip-gradient) lr-scheduler))
  ([]
   (nag {})))

(s/def ::rho number?)
(s/def ::rescale-gradient number?)
(s/def ::epsilon number?)
(s/def ::ada-delta-opts (s/keys :opt-un [::rho ::rescale-gradient ::epsilon ::wd ::clip-gradient]))

(defn ada-delta
  "AdaDelta optimizer as described in Matthew D. Zeiler, 2012.
   http://arxiv.org/abs/1212.5701"
  ([{:keys [rho rescale-gradient epsilon wd clip-gradient] :as opts
     :or {rho 0.05
          rescale-gradient 1.0
          epsilon 1e-8
          wd 0.0
          clip-gradient 0}}]
   (util/validate! ::ada-delta-opts opts "Incorrect ada-delta optimizer options")
   (new AdaDelta (float rho) (float rescale-gradient) (float epsilon) (float wd) (float clip-gradient)))
  ([]
   (ada-delta {})))

(s/def gamma1 number?)
(s/def gamma2 number?)
(s/def ::rms-prop-opts (s/keys :opt-un [::learning-rate ::rescale-gradient ::gamma1 ::gamma2 ::wd ::clip-gradient]))

(defn rms-prop
  "RMSProp optimizer as described in Tieleman & Hinton, 2012.
   http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
   - learningRate Step size.
   - gamma1  decay factor of moving average for gradient, gradient^^2.
   -  gamma2  momentum factor of moving average for gradient.
   -  rescale-gradient rescaling factor of gradient.
   -  wd L2 regularization coefficient add to all the weights
   -  clip-gradient clip gradient in range [-clip_gradient, clip_gradient]
   -  lr-scheduler The learning rate scheduler"
  ([{:keys [learning-rate rescale-gradient gamma1 gamma2 wd lr-scheduler clip-gradient] :as opts
     :or {learning-rate 0.002
          rescale-gradient 1.0
          gamma1 0.95
          gamma2 0.9
          wd 0.0
          clip-gradient 0}}]
   (util/validate! ::rms-prop-opts opts "Incorrect rms-prop optimizer options")
   (new RMSProp (float learning-rate) (float rescale-gradient) (float gamma1)
        (float gamma2) (float wd) lr-scheduler (float clip-gradient)))
  ([]
   (rms-prop {})))

(s/def ::ada-grad-opts (s/keys :opt-un [::learning-rate ::rescale-gradient ::epsilon ::wd]))

(defn ada-grad
  " AdaGrad optimizer as described in Duchi, Hazan and Singer, 2011.
   http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

   - learning-rate Step size.
   - epsilon A small number to make the updating processing stable.
                Default value is set to 1e-7.
   - rescale-gradient rescaling factor of gradient.
   - wd L2 regularization coefficient add to all the weights"
  ([{:keys [learning-rate rescale-gradient epsilon wd] :as opts
     :or {learning-rate 0.05
          rescale-gradient 1.0
          epsilon 1e-7
          wd 0.0}}]
   (util/validate! ::ada-grad-opts opts "Incorrect ada-grad optimizer options")
   (new AdaGrad (float learning-rate) (float rescale-gradient) (float epsilon) (float wd)))
  ([]
   (ada-grad {})))

(s/def ::beta1 number?)
(s/def ::beta2 number?)
(s/def ::adam-opts (s/keys :opt-un [::learning-rate ::beta1 ::beta2 ::epsilon ::decay-factor ::wd ::clip-gradient ::lr-scheduler]))

(defn adam
  "Adam optimizer as described in [King2014]

  [King2014] Diederik Kingma, Jimmy Ba,
  Adam: A Method for Stochastic Optimization,
  http://arxiv.org/abs/1412.6980

   - learning-rate  Step size.
   - beta1  Exponential decay rate for the first moment estimates.
   - beta2  Exponential decay rate for the second moment estimates.
   -  epsilon
   - decay-factor
   - wd L2 regularization coefficient add to all the weights
   - clip-gradient  clip gradient in range [-clip_gradient, clip_gradient]
   - lr-scheduler The learning rate scheduler"
  ([{:keys [learning-rate beta1 beta2 epsilon decay-factor wd clip-gradient lr-scheduler] :as opts
     :or {learning-rate 0.002
          beta1 0.9
          beta2 0.999
          epsilon 1e-8
          decay-factor (- 1 1e-8)
          wd 0
          clip-gradient 0}}]
   (util/validate! ::adam-opts opts "Incorrect adam optimizer options")
   (new Adam (float learning-rate) (float beta1) (float beta2) (float epsilon)
        (float decay-factor) (float wd) (float clip-gradient) lr-scheduler))
  ([]
   (adam {})))

(s/def ::sgld-opts (s/keys :opt-un [::learning-rate ::rescale-gradient ::wd ::clip-gradient ::lr-scheduler]))

(defn sgld
  "Stochastic Langevin Dynamics Updater to sample from a distribution.

  - learning-rate Step size.
  - rescale-gradient rescaling factor of gradient.
  - wd L2 regularization coefficient add to all the weights
  - clip-gradient Float, clip gradient in range [-clip_gradient, clip_gradient]
  - lr-scheduler The learning rate scheduler"
  ([{:keys [learning-rate rescale-gradient wd clip-gradient lr-scheduler] :as opts
     :or {learning-rate 0.01
          rescale-gradient 1
          wd 0.0001
          clip-gradient 0}}]
   (util/validate! ::sgld-opts opts "Incorrect sgld optimizer options")
   (new SGLD (float learning-rate) (float rescale-gradient) (float wd)
        (float clip-gradient) lr-scheduler))
  ([]
   (sgld {})))

(defn update
  "Update the parameters.
   - optimizer - the optimizer
   -  index An unique integer key used to index the parameters
   -  weight weight ndarray
   -  grad grad ndarray
    -  state NDArray or other objects returned by initState
             The auxiliary state used in optimization.
  "
  [optimizer index weight grad state]
  (doto optimizer
    (.update (int index) weight grad state)))

(defn create-state
  "Create additional optimizer state such as momentum."
  [optimizer index weight]
  (do
    (.createState optimizer (int index) weight)))

