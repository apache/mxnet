# POS_Trainer is the stage one : partition optmizer status in DeepSpeed's work
# It can reduce memory consumption in distributed Trainer but slower
# since we can not solve overlapping problem when calling broadcast and optimize parameters.
# The usage of this trainer is totally same with original one
# I test some benchmark Here:
# For 4 V100 Gpu with 16GB memory, the maximum batch size for bert-large and bert-base:
# bert-large: Original: 16 Pos: 24
# bert-base: Original: 64 Pos: 80
# The ideal average saving memory for each GPU is: (N-1)/N * P * K
# where N is the GPU number, P is the parameter number and K is the memory
# multiplier of optimizer states(E.g. for Adam, K = 12)
#TODO add group_num
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size,  rank, local_rank
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

import mxnet as mx
from collections import OrderedDict, defaultdict
import types
import time
import warnings
from mxnet.gluon.parameter import Parameter
from horovod.mxnet.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set

class _NCCLReduceHelper(object):
    _init = False
    nccl_id = None
    num_gpus = None
    rank = None

    @staticmethod
    def init(num_gpus, root_rank):
        """Communicate the NCCL unique id"""
        cls = _NCCLReduceHelper
        if not cls._init:
            cls._init = True
            import ctypes
            try:
                from mpi4py import MPI
            except:
                raise ImportError("Spatial parallel modules require mpi4py package.")
            import numpy as np
            nccl_id_size = ctypes.c_int()
            check_call(_LIB.MXNCCLGetUniqueIdSize(ctypes.byref(nccl_id_size)))
            nccl_id_size = nccl_id_size.value
            cls.nccl_id = np.zeros(nccl_id_size, np.byte)
            check_call(_LIB.MXNCCLGetUniqueId(
                cls.nccl_id.ctypes.data_as(ctypes.c_void_p)))
            global_comm = MPI.COMM_WORLD
            rank = global_comm.rank
            color = rank / num_gpus
            comm = global_comm.Split(color, rank)
            comm.Bcast([cls.nccl_id, nccl_id_size, MPI.BYTE], root=0)
            cls.num_gpus = num_gpus
            cls.rank = rank % num_gpus
            cls.root_rank = root_rank % num_gpus
        assert num_gpus == cls.num_gpus


class POS_Trainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None,
                 gradient_predivide_factor=1.0, prefix=None, partition_gradients = False):

        self._world_size = size()
        self._world_rank = rank()

        self._partition_gradients = partition_gradients

        self._all_params = []
        self._all_param2idx = {}
        self._all_params_with_names = params
        param_list = []
        if isinstance(params, (dict, OrderedDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s." % (type(params)))
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s." % (type(param)))
            if param._uuid in self._all_param2idx:
                # Shared parameters have same uuid; only need to store one of the shared versions
                continue
            self._all_param2idx[param._uuid] = i
            self._all_params.append(param)
        self._partition_params, self._param2rank = self._partition_parameters(self._all_params)
        self._own_part = self._partition_params[self._world_rank]
        super(POS_Trainer, self).__init__(
            self._own_part, optimizer, optimizer_params=optimizer_params, kvstore=None)
        self._prefix = prefix if prefix else ""
        self._scale = gradient_predivide_factor / size()
        self._gradient_predivide_factor = gradient_predivide_factor



    def _partition_parameters(self, params):
        """
        partition all the parameters by their size and try to average them.
        """
        world_size = self._world_size
        ## list for rank each would be
        partition_params = [[] for _ in range(world_size)]
        param2rank = {}
        sizes = [0 for _ in range(world_size)]
        for param in params:
            if param.grad_req != 'null':
                current_rank = sizes.index(min(sizes))
                partition_params[current_rank].append(param)
                num = 1
                param2rank[param._uuid] = current_rank
                for p in param.shape:
                    num *= p
                sizes[current_rank] += num
        return partition_params, param2rank

    def _allreduce_grads(self):
        """
        rewrite allreduce here because we need to communicate using horovod.
        Actually we should use reduce here, but since it is not available yet,
        I use allreduce instead.
        """
        if not self._partition_gradients:
            for i, param in enumerate(self._all_params):
                if param.grad_req != 'null':
                    allreduce_(param.list_grad()[0], average=False,
                               name=self._prefix + str(i), priority=-i,
                               prescale_factor=1.0 / self._gradient_predivide_factor)





    def step(self, batch_size, ignore_stale_grad=False):
        """
        inherit from trainer, only call boardcast to make sure all parameter are consistent
        Makes one step of parameter update.
        Since each process main their own part, we need to brodcast after calculation
        """
        super(POS_Trainer, self).step(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

    def update(self, batch_size, ignore_stale_grad=False):
        '''
        assert not (self._kvstore and self._update_on_kvstore), \
            'update() when parameters are updated on kvstore ' \
            'is not supported. Try setting `update_on_kvstore` ' \
            'to False when creating trainer.'
        Since each process main their own part, we need to brodcast after calculation
        '''


        super(POS_Trainer, self).update(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()

    def _broadcast_partition_params(self):
        """
        This function is to broadcast parameter since each process will maintain their own part
        """
        for param in self._all_params:
            broadcast_(param.data(), self._param2rank[param._uuid], name=str(self._all_param2idx[param._uuid]))

    def correspond_ranks(self):
        return self._param2rank

    def generate_graph_pass_options(self):
        #helper = _NCCLReduceHelper
        #helper.init(size(), 0)
        #helper2 = _NCCLReduceHelper
        #helper2.init(size(), 0)
        options = {}
        for name in self._all_params_with_names:
            type = name.split('.')[-1]
            index = self._param2rank[self._all_params_with_names[name]._uuid]
            new_name = self._all_params_with_names[name]._uuid.replace('-', '_') + '_' + type
            options[new_name] = index

        helper = _NCCLReduceHelper
        helper.init(size(), 0)
        options.update({"num_gpus": size(), "rank": rank(), "nccl_unique_id":helper.nccl_id.ctypes.data})
        return options

    def generate_backward_options(self):
        backward_option = {"partition_grad":True, "current_rank":rank()}
        for name in self._all_params_with_names:
            type = name.split('.')[-1]
            index = self._param2rank[self._all_params_with_names[name]._uuid]
            new_name = 'ncclreduce_' + self._all_params_with_names[name]._uuid.replace('-', '_') + '_' + type + "_backward"
            backward_option[new_name] = index
        return backward_option
