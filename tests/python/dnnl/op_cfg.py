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

from collections import namedtuple
from itertools import product
from functools import partial

import mxnet as mx
from mxnet.base import (_is_np_op, _NP_OP_PREFIX, _NP_EXT_OP_PREFIX, _NP_INTERNAL_OP_PREFIX,
                        _OP_NAME_PREFIX_LIST)

PREFIX_TO_MODULE = {
    _NP_OP_PREFIX: mx.sym.np,
    _NP_INTERNAL_OP_PREFIX: mx.sym.np._internal,
    _NP_EXT_OP_PREFIX: mx.sym.npx
}
for nd_prefix in _OP_NAME_PREFIX_LIST:
    module_name = nd_prefix[1:-1]  # nd_prefix == '_<module_name>_'
    PREFIX_TO_MODULE[nd_prefix] = getattr(mx.sym, module_name)

CFG_BASED_ON = '__based_on__'
CFG_SUBGRAPH = '__subgraph__'
CFG_RTOL_ATOL = '__rtol_atol__'
DEFAULT_SHAPE = (8,)

TensorArg = namedtuple('TensorArg', ['gen_tensor'])
CfgBasedArg = namedtuple('CfgBasedArg', ['gen_arg'])
SubgraphCfg = namedtuple('SubgraphCfg', ['base_op', 'backend'])


def get_op_sym_fn(op_name: str):
    for prefix, module in PREFIX_TO_MODULE.items():
        if op_name.startswith(prefix):
            return getattr(module, op_name[len(prefix):])
    try:
        return getattr(mx.sym, op_name)
    except AttributeError:
        try:
            # op with '_' prefix
            return getattr(mx.sym, op_name[1:])
        except AttributeError:
            return getattr(mx.sym._internal, op_name)


def default_tensor(dim_or_shape, dtype):
    if isinstance(dim_or_shape, (tuple, list)):
        shape = dim_or_shape
        return mx.nd.random.normal(0, 1, shape, dtype)
    dim = dim_or_shape
    return mx.nd.random.normal(0, 1, DEFAULT_SHAPE*dim, dtype)


def common_weight_tensor(shape, dtype, numpy=False):
    tensor = mx.nd.random.normal(0, 0.1, shape, dtype)
    return tensor.as_np_ndarray() if numpy else tensor


def valatt_attention_tensor(cfg):
    qkv = cfg['queries_keys_values']
    batch, seq_len, _ = qkv.shape
    heads = cfg['heads']
    att_shape = (batch, heads, seq_len, seq_len)
    return mx.nd.random.randint(0, 2, att_shape).astype(qkv.dtype)


def get_all_ops_cfgs(dtype):
    return {
        'Convolution': {
            'data,kernel': [
                (default_tensor(3, dtype), (3,)),
                (default_tensor(4, dtype), (3, 3)),
                (default_tensor(5, dtype), (3, 3, 3))
            ],
            'weight': [TensorArg(common_weight_tensor)],
            'bias': [TensorArg(common_weight_tensor)],
            'no_bias': [False],
            'num_filter': [8]
        },
        'Deconvolution': {CFG_BASED_ON: 'Convolution'},
        'FullyConnected': {
            'data': [default_tensor(2, dtype)],
            'weight': [TensorArg(common_weight_tensor)],
            'bias': [TensorArg(common_weight_tensor)],
            'num_hidden': [3]
        },
        'Pooling': {
            'data,kernel': [
                (default_tensor(3, dtype), (3,)),
                (default_tensor(4, dtype), (3, 3)),
                (default_tensor(5, dtype), (3, 3, 3))
            ],
        },
        '_contrib_AdaptiveAvgPooling2D': {
            'data,kernel,output_size': [(default_tensor(4, dtype), (2, 2), (4, 4))],
        },

        ######################################### Casting #########################################

        'Cast': {
            'data': [default_tensor(2, dtype)],
            'dtype': ['bool']
        },
        '_contrib_quantize_v2': {
            'data': [default_tensor(2, dtype)],
            'min_calib_range': [CfgBasedArg(lambda cfg: cfg['data'].min().asscalar())],
            'max_calib_range': [CfgBasedArg(lambda cfg: cfg['data'].max().asscalar())],
            CFG_RTOL_ATOL: [(0, 0)]
        },

        ##################################### No calculations #####################################

        'Flatten': {
            'data': [default_tensor(2, dtype)]
        },
        'Concat': {
            '0,1,dim': [
                (default_tensor(2, dtype), default_tensor(2, dtype), 0)
            ]
        },
        'Reshape': {
            '0': [default_tensor(2, dtype)],
            '1': [(-1,)]
        },
        'transpose': {
            '0': [default_tensor(2, dtype)]
        },
        'expand_dims': {
            'data': [default_tensor(2, dtype)],
            'axis': [-1]
        },
        'where': {
            'x,y,condition': [
                (default_tensor(2, dtype),
                 default_tensor(2, dtype),
                 mx.nd.random.randint(0, 2, DEFAULT_SHAPE*2, 'int32'))
            ],
        },
        'take': {
            'a,indices': [
                (default_tensor(2, dtype),
                 mx.nd.random.randint(0, DEFAULT_SHAPE[0], (2,), 'int32'))
            ],
            'axis': [-1]
        },
        'stack': {
            '0,1': [
                (default_tensor(2, dtype), default_tensor(2, dtype))
            ]
        },
        '_split_v2': {
            'ary,indices_or_sections': [
                (default_tensor(2, dtype), (2, 3))
            ],
        },
        'slice': {
            'data,begin,end': [
                (default_tensor(2, dtype), (0, 1), (2, 4))
            ],
        },
        'space_to_depth': {
            'data,block_size': [
                (default_tensor(4, dtype), 2)
            ],
        },
        '_copy': {
            'data': [default_tensor(2, dtype)]
        },
        '_npi_transpose': {CFG_BASED_ON: 'transpose'},
        '_npi_where': {CFG_BASED_ON: 'where'},
        '_npx_reshape': {CFG_BASED_ON: 'Reshape'},

        ###################################### Normalization ######################################

        'LayerNorm': {
            'data': [default_tensor(2, dtype)],
            'gamma': [TensorArg(common_weight_tensor)],
            'beta': [TensorArg(common_weight_tensor)],
        },
        'BatchNorm': {
            CFG_BASED_ON: 'LayerNorm',
            'moving_mean': [TensorArg(common_weight_tensor)],
            'moving_var': [
                TensorArg(lambda shape, dtype: mx.nd.random.uniform(0, 1, shape, dtype))
            ],
        },
        'LRN': {
            'data,nsize': [(default_tensor(2, dtype), 3)]
        },

        ######################################## Reduction ########################################

        'mean': {
            '0': [default_tensor(2, dtype)],
            'axis': [0]
        },
        'sum': {CFG_BASED_ON: 'mean'},
        '_npi_mean': {CFG_BASED_ON: 'mean'},
        '_npi_sum': {CFG_BASED_ON: 'mean'},

        ######################################### Softmax #########################################

        'softmax': {
            'data': [
                default_tensor(2, dtype),
                default_tensor(4, dtype)
            ],
            'axis': [-1]
        },
        'log_softmax': {CFG_BASED_ON: 'softmax'},
        'masked_softmax': {
            CFG_BASED_ON: 'softmax',
            'mask': [
                CfgBasedArg(
                    lambda cfg: mx.nd.random.randint(0, 2, cfg['data'].shape).astype('bool')
                )
            ],
        },

        ################################### Activation / Unary ####################################

        'Activation': {
            'data': [default_tensor(2, dtype)],
            'act_type': ['sigmoid', 'log_sigmoid', 'relu', 'softrelu', 'tanh', 'mish']
        },
        'LeakyReLU': {
            'data': [default_tensor(2, dtype)],
            'act_type': ['leaky', 'elu', 'gelu']
        },
        '_npi_exp': {
            '0': [default_tensor(2, dtype)]
        },
        '_npi_sqrt': {
            '0': [mx.nd.random.uniform(0, 8, DEFAULT_SHAPE*2, dtype)]
        },
        '_npi_square': {CFG_BASED_ON: '_npi_exp'},
        '_npi_tanh': {CFG_BASED_ON: '_npi_exp'},

        ######################################### Binary ##########################################

        'dot': {
            '0,1': [
                (default_tensor(3, dtype), default_tensor(3, dtype))
            ],
        },
        'batch_dot': {CFG_BASED_ON: 'dot'},
        'broadcast_add': {CFG_BASED_ON: 'dot'},
        'broadcast_div': {CFG_BASED_ON: 'dot'},
        'broadcast_mul': {CFG_BASED_ON: 'dot'},
        'broadcast_sub': {CFG_BASED_ON: 'dot'},
        'elemwise_add': {CFG_BASED_ON: 'dot'},
        '_npi_dot': {CFG_BASED_ON: 'dot'},
        '_npi_add': {CFG_BASED_ON: 'dot'},
        '_npi_multiply': {CFG_BASED_ON: 'dot'},
        '_npi_subtract': {CFG_BASED_ON: 'dot'},
        '_npi_true_divide': {CFG_BASED_ON: 'dot'},

        'add_n': {CFG_BASED_ON: 'dot'},  # this is not binary, but can work as binary

        ######################################## Subgraph #########################################

        '_sg_onednn_conv': {
            CFG_BASED_ON: 'Convolution',
            CFG_SUBGRAPH: [SubgraphCfg('Convolution', 'ONEDNN')],
            'data,kernel': [
                (default_tensor(4, dtype), (3, 3)),
                (default_tensor(5, dtype), (3, 3, 3))
            ]
        },
        '_sg_onednn_fully_connected': {
            CFG_BASED_ON: 'FullyConnected',
            CFG_SUBGRAPH: [SubgraphCfg('FullyConnected', 'ONEDNN')],
        },
        '_sg_onednn_batch_dot': {
            CFG_BASED_ON: 'batch_dot',
            CFG_SUBGRAPH: [SubgraphCfg('batch_dot', 'ONEDNN')],
        },
        '_sg_onednn_batch_norm': {CFG_BASED_ON: 'BatchNorm'},
        '_sg_onednn_selfatt_qk': {
            CFG_SUBGRAPH: [SubgraphCfg('_sg_onednn_selfatt_qk', 'ONEDNN')],
            'queries': [mx.nd.random.normal(0, 1, (1, 4, 3*2*8), dtype)],
            'keys': [mx.nd.random.normal(0, 1, (1, 8, 3*2*8), dtype)],
            'heads': [2]
        },
        '_sg_onednn_selfatt_qk_split': {
            CFG_SUBGRAPH: [SubgraphCfg('_sg_onednn_selfatt_qk_split', 'ONEDNN')],
            'queries_keys_values': [mx.nd.random.normal(0, 1, (1, 4, 3*2*8), dtype)],
            'heads': [2]
        },
        '_sg_onednn_selfatt_valatt': {
            CFG_BASED_ON: '_sg_onednn_selfatt_qk_split',
            CFG_SUBGRAPH: [SubgraphCfg('_sg_onednn_selfatt_valatt', 'ONEDNN')],
            'attention': [CfgBasedArg(valatt_attention_tensor)]
        }
    }


def product_dict(dict_of_lists):
    keys = dict_of_lists.keys()
    lists = dict_of_lists.values()
    for scenario in product(*lists):
        yield dict(zip(keys, scenario))


def resolve_cfg_references(args_cfg, all_ops_cfgs):
    if len(args_cfg) == 0:
        return {}
    args_cfg = args_cfg.copy()
    base_op = args_cfg.pop(CFG_BASED_ON, None)
    base_cfg = all_ops_cfgs.get(base_op, {})
    result_cfg = resolve_cfg_references(base_cfg, all_ops_cfgs)
    result_cfg.update(args_cfg)
    return result_cfg


def get_op_cfg_generator(op_names, dtype):
    all_ops_cfgs = get_all_ops_cfgs(dtype)
    for op_name in set(op_names):
        args_cfgs = all_ops_cfgs[op_name]
        args_cfgs = resolve_cfg_references(args_cfgs, all_ops_cfgs)
        for args_scenario in product_dict(args_cfgs):
            yield (op_name, args_scenario)


def get_symblock_from_args_scenario(op_name, args_scenario):
    args_scenario = args_scenario.copy()
    args_scenario.pop(CFG_RTOL_ATOL, None)  # not used here
    subgraph_cfg = args_scenario.pop(CFG_SUBGRAPH, None)
    if subgraph_cfg is None:
        op_sym_fn = get_op_sym_fn(op_name)
    else:
        op_sym_fn = get_op_sym_fn(subgraph_cfg.base_op)


    # split binded args
    binded_args = [(k, v) for k, v in args_scenario.items() if ',' in k]
    for arg_names, arg_cfgs in binded_args:
        args_scenario.pop(arg_names)
        arg_names = arg_names.replace(' ', '').split(',')
        assert isinstance(arg_cfgs, tuple) and len(arg_cfgs) == len(arg_names)
        for arg_name, arg_cfg in zip(arg_names, arg_cfgs):
            assert arg_name not in args_scenario
            args_scenario[arg_name] = arg_cfg

    # generate cfg based args
    for arg_name, arg_cfg in args_scenario.items():
        if isinstance(arg_cfg, CfgBasedArg):
            args_scenario[arg_name] = arg_cfg.gen_arg(args_scenario)

    kw_args = {}
    pos_args = {}
    for arg_name, arg_cfg in args_scenario.items():
        if isinstance(arg_cfg, (TensorArg, mx.nd.NDArray, mx.np.ndarray)):
            arg_cfg = mx.sym.var(arg_name)
            if _is_np_op(op_name):
                arg_cfg = arg_cfg.as_np_ndarray()
        if arg_name.isdigit():
            pos_args[int(arg_name)] = arg_cfg
        else:
            kw_args[arg_name] = arg_cfg
    pos_args = [pos_args[k] for k in sorted(pos_args.keys())]

    sym = op_sym_fn(*pos_args, **kw_args)
    if subgraph_cfg is not None:
        if len(sym.list_outputs()) > 1:
            sym = sym[0]
        # add additional op (+1), so the graph pass can convert the tested op
        sym = mx.sym.relu(sym).optimize_for(subgraph_cfg.backend)
    assert op_name in sym.tojson()

    args_with_shape, args_with_dtype = {}, {}
    for arg_name, arg_cfg in args_scenario.items():
        if isinstance(arg_cfg, (mx.nd.NDArray, mx.np.ndarray)):
            args_with_shape[arg_name] = arg_cfg.shape
            args_with_dtype[arg_name] = arg_cfg.dtype

    infered_shapes_args, _, infered_shapes_auxs = sym.infer_shape(**args_with_shape)
    infered_shapes_args = dict(zip(sym.list_arguments(), infered_shapes_args))
    infered_shapes_auxs = dict(zip(sym.list_auxiliary_states(), infered_shapes_auxs))

    infered_dtypes_args, _, infered_dtypes_auxs = sym.infer_type(**args_with_dtype)
    infered_dtypes_args = dict(zip(sym.list_arguments(), infered_dtypes_args))
    infered_dtypes_auxs = dict(zip(sym.list_auxiliary_states(), infered_dtypes_auxs))

    symblock_input_data = {}
    for arg_name in [*sym.list_arguments(), *sym.list_auxiliary_states()]:
        tensor_cfg = args_scenario[arg_name]
        if isinstance(tensor_cfg, TensorArg):
            shape = infered_shapes_args.get(arg_name, infered_shapes_auxs.get(arg_name, None))
            dtype = infered_dtypes_args.get(arg_name, infered_dtypes_auxs.get(arg_name, None))
            tensor = tensor_cfg.gen_tensor(shape, dtype)
        else:
            tensor = tensor_cfg
        symblock_input_data[arg_name] = tensor

    symblock_input_syms = [mx.sym.var(name) for name in symblock_input_data.keys()]
    if _is_np_op(op_name):
        symblock_input_syms = [var.as_np_ndarray() for var in symblock_input_syms]
    symblock = mx.gluon.SymbolBlock(sym, symblock_input_syms)
    symblock.initialize()
    assert len(symblock.collect_params()) == 0

    return symblock, list(symblock_input_data.values())
