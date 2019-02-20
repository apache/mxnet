from ..context import cpu
from ..ndarray import multi_all_finite
from ..ndarray import ndarray as nd
from math import ceil
from .. import autograd as ag

class LossScaler(object):
    def __init__(self):
        self._loss_scale = 2.**16
        self._next_loss_scale = self._loss_scale
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._has_overflow = False

    @property
    def loss_scale(self):
        return self._loss_scale

    def launch_check_overflow(self, params):
        self._wait_for_outputs = True
        self._has_overflow = False
        with ag.pause():
            chunk_size = 200
            valid_params = [p._grad[0] for p in params if p._grad is not None]
            gpu_output = nd.ones((1,), ctx=valid_params[0].context)
            nb_params = len(valid_params)
            [multi_all_finite(*valid_params[idx:idx+chunk_size],
                              num_arrays=len(valid_params[idx:idx+chunk_size]),
                              init_output=False, out=gpu_output) for idx in range(0, nb_params,
                              chunk_size)]
            self.output = gpu_output.as_in_context(cpu())
        #with ag.pause():
        #    valid_params = [p._grad for p in params if p._grad is not None]
        #    #self.outputs = [nd.ones((1,), ctx=g.context) for g in valid_params[0]]
        #    gpu_outputs = [nd.ones((1,), ctx=g.context) for g in valid_params[0]]
        #    nb_ctx = len(gpu_outputs)
        #    self.outputs = nd.ones((nb_ctx,), ctx=cpu())
        #    nb_params = len(valid_params)
        #    nb_params_per_ctx = ceil(nb_params / float(nb_ctx))
        #    nb_cycle = ceil(nb_params / (nb_ctx * 200))
        #    params_chunk = [[g[ctx_idx] for g in valid_params[ctx_idx * nb_params_per_ctx:(ctx_idx+1) * nb_params_per_ctx]] for ctx_idx in range(nb_ctx)]
        #    offset = 0
        #    for n in range(nb_cycle):
        #        [multi_all_finite(*params_chunk[ctx_idx][offset:offset+200],
        #                         num_arrays=len(params_chunk[ctx_idx][offset:offset+200]),
        #                         out=gpu_outputs[ctx_idx]) for ctx_idx in range(nb_ctx)]
        #        offset += 200
        #    for i, out in enumerate(gpu_outputs):
        #        out.copyto(self.outputs[i])

    def wait_and_update(self):
        if self._wait_for_outputs:
            #np_out = self.outputs.asnumpy()
            #self._has_overflow = not np_out.astype('bool').all()
            self._has_overflow = not bool(self.output.asnumpy())
            self._loss_scale = self._next_loss_scale
            if self._has_overflow:
                self._next_loss_scale = self._loss_scale / 2.
                self._unskipped = 0
                print("_HAS_OVERFLOW")
                print("loss scale is %f, but will be %f next iteration" % (self._loss_scale, self._next_loss_scale))
            else:
                self._unskipped += 1
            if self._unskipped == self._scale_seq_len:
                self._unskipped = 0
                self._next_loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
                print("SCALE UP")
                print("loss scale is %f, but will be %f next iteration" % (self._loss_scale, self._next_loss_scale))
            self._wait_for_outputs = False
        return self._has_overflow
