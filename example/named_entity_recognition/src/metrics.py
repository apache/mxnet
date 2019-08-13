# !/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

tag_dict = load_obj("../data/tag_to_index")
not_entity_index = tag_dict["O"]

def classifer_metrics(label, pred):
    """
    computes f1, precision and recall on the entity class
    """
    prediction = np.argmax(pred, axis=1)
    label = label.astype(int)

    pred_is_entity = prediction != not_entity_index
    label_is_entity = label != not_entity_index

    corr_pred = (prediction == label) == (pred_is_entity == True)

    #how many entities are there?
    num_entities = np.sum(label_is_entity)
    entity_preds = np.sum(pred_is_entity)

    #how many times did we correctly predict an entity?
    correct_entitites = np.sum(corr_pred[pred_is_entity])

    #precision: when we predict entity, how often are we right?
    if entity_preds == 0:
        precision = np.nan
    else:
        precision = correct_entitites/entity_preds

    #recall: of the things that were an entity, how many did we catch?
    recall = correct_entitites / num_entities
    if num_entities == 0:
        recall = np.nan
    # To prevent dozens of warning: RuntimeWarning: divide by zero encountered in long_scalars
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def entity_precision(label, pred):
    return classifer_metrics(label, pred)[0]

def entity_recall(label, pred):
    return classifer_metrics(label, pred)[1]

def entity_f1(label, pred):
    return classifer_metrics(label, pred)[2]

def composite_classifier_metrics():
    metric1 = mx.metric.CustomMetric(feval=entity_precision, name='entity precision')
    metric2 = mx.metric.CustomMetric(feval=entity_recall, name='entity recall')
    metric3 = mx.metric.CustomMetric(feval=entity_f1, name='entity f1 score')
    metric4 = mx.metric.Accuracy()

    return mx.metric.CompositeEvalMetric([metric4, metric1, metric2, metric3])
