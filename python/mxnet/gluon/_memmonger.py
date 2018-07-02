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

# coding: utf-8
# pylint: disable=

import math

from .. import cpu

def prod(shape):
    """Get product of the shape.
    """
    ret = 1
    for s in shape:
        ret *= s
    return ret

def make_mirror_plan(sym, inputs, params, threshold, plan_info=None):
    """Memory allocation planner with a given threshold.
    The user can pass in a network configuration,
    a threshold that limits memory per block.
    And input shape configurations.
    Parameters
    ----------
    sym : symbol
        Input configuration of symbols.
        The user need to pre-mark the attribute "mirror_stage" on the nodes
        that can be book-kept as stage
        The algorithm will decide whether to disbale mirror on the stage nodes.
    threshold: integer
        A tuning parameter to tune the approximate size of each stage blocks
    plan_info: dict, optional
        Used to hold plan information.
    **kwargs:
        The arguments to infer shape.
    Returns
    -------
    alloc_sym: symbol
        A symbol with force mirror tagged on the nodes for better allocation.
    """
    threshold = threshold << 20
    sym = sym.__copy__()
    internals = sym.get_internals()
    input_shapes = {key: val.shape for key, val in inputs.items()}
    input_shapes.update({key: val.shape for key, val in params.items()})
    _, out_shapes, _ = internals.infer_shape(**input_shapes)
    shape_dict = list(zip(internals.list_outputs(), out_shapes))
    total_size = 0
    local_size = 0
    save_size = 0
    max_size = 0
    last_sb = None
    last_local = 0
    period = 1
    last_stage = ''
    stage_decision = ''

    for idx, item in enumerate(shape_dict):
        sb = internals[idx]
        name, shape = item
        if name in input_shapes:
            continue
        total_size += prod(shape) * 4
        local_size += prod(shape) * 4
        sb._set_attr(force_mirroring='True')

        if sb.attr('mirror_stage') is not None:
            stage = sb.attr('mirror_stage')
            if stage == 'True' or stage != last_stage:
                if local_size > threshold:
                    save_size += prod(shape) * 4
                    max_size = max(max_size, local_size)
                    local_size = 0
                    stage_decision = 'False'
                    sb._set_attr(force_mirroring=stage_decision)
                else:
                    stage_decision = 'True'
                    pass
                last_stage = stage
            elif stage == last_stage and stage_decision == 'False':
                save_size += prod(shape) * 4
                sb._set_attr(force_mirroring=stage_decision)

    if plan_info is not None:
        plan_info['max_size'] = max_size
        plan_info['save_size'] = save_size
    return sym


def get_cost(sym, inputs, params):
    """Get the cost of the current symbolic plan by running bind on CPU.
    sym : Symbolic Variable
    """
    grad_reqs = {}
    type_dict = {}
    shape_dict = {}
    for key, val in inputs.items():
        type_dict[key] = val.dtype
        shape_dict[key] = val.shape
        if val.grad is None:
            grad_reqs[key] = 'null'
        else:
            grad_reqs[key] = 'write'

    for key, val in params.items():
        type_dict[key] = val.dtype
        shape_dict[key] = val.shape
        grad_reqs[key] = val.grad_req

    texec = sym.simple_bind(ctx=cpu(),
                            grad_req=grad_reqs,
                            type_dict=type_dict,
                            **shape_dict)
    return int(texec.debug_str().split('\n')[-3].split()[1])


def search_plan(sym, inputs, params, ntrials=6):
    """Quickly heurestic search over possible plans to find good memory plan.

    Parameters
    ----------
    sym : symbolic
       Symbolic configurations
    ntrials: integer
       Additional grid search steps
    """
    history = []
    threshold = 0
    min_threshold = None
    min_cost = None
    nbegin = 3

    for k in range(nbegin):
        info = {}
        sym = make_mirror_plan(sym, inputs, params, threshold, info)
        cost = get_cost(sym, inputs, params)
        save_size = info['save_size'] >> 20
        local_size = info['max_size'] >> 20
        guess = int(math.sqrt(save_size * local_size / 2))
        if min_cost is None or min_cost > cost:
            min_cost = cost
        if min_threshold is None or local_size < min_threshold:
            min_threshold = local_size
        print ("Search threshold=%d MB, cost=%d MB" % (threshold, cost))
        history.append((cost, threshold, sym))
        threshold = guess

    max_threshold = threshold * math.sqrt(2)
    step = int((max_threshold - min_threshold) / ntrials)
    threshold = min_threshold + step
    if step > 0:
        for k in range(ntrials):
            sym = make_mirror_plan(sym, inputs, params, threshold)
            cost = get_cost(sym, inputs, params)
            print ("Search threshold=%d MB, cost=%d MB" % (threshold, cost))
            history.append((cost, threshold, sym))
            threshold += step

    history.sort(key = lambda x: x[0])
    cost, threshold, sym = history[0]
    print('Find best plan with threshold=%d, cost=%d MB' % (threshold, cost))
    return sym
