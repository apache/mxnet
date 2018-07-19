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

import mxnet as mx


def load_param(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v.as_in_context(ctx)
        if tp == 'aux':
            aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params


def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict


def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict


def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)


def initialize_frcnn(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
    arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
    arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
    arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
    arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
    arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
    arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
    arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
    arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
    arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
    return arg_params, aux_params


def get_fixed_params(symbol, fixed_param_prefix=''):
    fixed_param_names = []
    if fixed_param_prefix:
        for name in symbol.list_arguments():
            for prefix in fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)
    return fixed_param_names
