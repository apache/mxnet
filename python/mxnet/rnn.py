from collections import namedtuple, deque
from copy import copy

from . import symbol
from . import metric
from . import context
from . import ndarray as nd

from .ndarray import NDArray, zeros
from .optimizer import get_updater
from .initializer import Uniform

# TODO: those are to be stored in the symbols like attributes
# connecting output delayed by t steps to input
RecurrentRule = namedtuple('RecurrentRule', ['output', 'input', 't'])

def SimpleRecurrece(data, name, num_hidden, act_type='relu'):
    recurrent_name = name + '_recurrent'
    recurrent_var  = symbol.Variable(recurrent_name)
    recurrent_hid  = symbol.FullyConnected(data=recurrent_var, name=name+'_rhid', num_hidden=num_hidden, no_bias=True)
    input_hid      = symbol.FullyConnected(data=data, name=name+'_ihid', num_hidden=num_hidden)
    output         = symbol.Activation(data=input_hid+recurrent_hid, name=name, act_type=act_type)
    output_name    = name + '_output'

    return (output, [RecurrentRule(output=output_name, input=recurrent_name, t=1)])


class Sequencer(object):
    def __init__(self, sym, rules, ctx, arg_params=None, aux_params=None, initializer=Uniform(0.01)):
        self.orig_sym = sym
        self.rules    = rules
        self.sym      = self.extract_states(sym, rules)
        self.ctx      = ctx

        self.arg_params = arg_params
        self.aux_params = aux_params
        self.initializer = initializer

    def infer_shape(self, **data_shapes):
        arg_shapes, out_shapes, aux_shapes = self.sym.infer_shape_partial(**data_shapes)
        out_names = self.sym.list_outputs()
        for rule in self.rules:
            data_shapes[rule.input] = out_shapes[out_names.index(rule.output)]
        return self.sym.infer_shape(**data_shapes)

    @property
    def state_names(self):
        return [x.input for x in self.rules]

    @property
    def max_delay(self):
        return max([x.t for x in self.rules])

    def _get_param_names(self, input_names):
        arg_names   = self.sym.list_arguments()
        param_names = list(set(arg_names) - set(input_names) - set(self.state_names))
        return param_names

    def _init_params(self, input_shapes, overwrite=False):
        arg_shapes, out_shapes, aux_shapes = self.infer_shape(**input_shapes)
        #for n, s in zip(self.sym.list_arguments(), arg_shapes):
        #    print('%s: %s' % (n, s))

        arg_names   = self.sym.list_arguments()
        input_names = input_shapes.keys()
        aux_names   = self.sym.list_auxiliary_states()
        param_names = self._get_param_names(input_names)

        param_name_shapes = [x for x in zip(arg_names, arg_shapes) if x[0] in param_names]
        arg_params = {k: nd.zeros(s) for k, s in param_name_shapes}
        aux_params = {k: nd.zeros(s) for k, s in zip(aux_names, aux_shapes)}

        for k, v in arg_params.items():
            if self.arg_params and k in self.arg_params and (not overwrite):
                arg_params[k][:] = self.arg_params[k]
            else:
                print("Initializing %s" % k)
                self.initializer(k, v)

        for k, v in aux_params.items():
            if self.aux_params and k in self.aux_params and (not overwrite):
                aux_params[k][:] = self.aux_params[k]
            else:
                self.initializer(k, v)

        self.arg_params = arg_params
        self.aux_params = aux_params
        return (arg_names, param_names, aux_names)

    def bind_executor(self, ctx, need_grad=False, **input_shapes):
        arg_shapes, out_shapes, aux_shapes = self.infer_shape(**input_shapes)
        arg_names = self.sym.list_arguments()
        param_names = self._get_param_names(input_shapes.keys())
        state_names = self.state_names

        arg_arrays = [zeros(s) for s in arg_shapes]
        aux_arrays = [zeros(s) for s in aux_shapes]
        grad_req   = ['null' for x in arg_arrays]
        for i in range(len(arg_names)):
            if arg_names[i] in param_names:
                grad_req[i] = 'add'
            elif arg_names[i] in state_names:
                grad_req[i] = 'write'

        grad_arrays = {arg_names[i]: zeros(arg_shapes[i]) \
                for i in range(len(arg_names)) if grad_req[i] != 'null'}

        return self.sym.bind(ctx, arg_arrays, grad_arrays, grad_req, aux_arrays)


    def fit(self, data, optimizer, eval_data=None, begin_epoch=0, end_epoch=1, eval_metric='acc'):
        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        input_shapes = dict(data.provide_data+data.provide_label)
        arg_names, param_names, aux_names = self._init_params(input_shapes)
        state_names = self.state_names
        data_names = [x[0] for x in data.provide_data]
        label_names = [x[0] for x in data.provide_label]
        output_names = self.sym.list_outputs()

        param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        state_idx = [i for i in range(len(arg_names)) if arg_names[i] in state_names]

        n_loss = len(self.orig_sym.list_outputs())
        n_state = len(self.sym.list_outputs()) - n_loss
        out_state_idx = range(n_loss, n_state+n_loss)

        exec_train = self.bind_executor(self.ctx, need_grad=True, **input_shapes)
        exec_train.copy_params_from(self.arg_params, self.aux_params)

        param_arrays = [exec_train.arg_arrays[i] for i in param_idx]
        state_arrays = [exec_train.arg_arrays[i] for i in state_idx]
        data_arrays  = [exec_train.arg_dict[name] for name in data_names]
        label_arrays = [exec_train.arg_dict[name] for name in label_names]

        param_grads  = [exec_train.grad_arrays[i] for i in param_idx]
        state_grads  = [exec_train.grad_arrays[i] for i in state_idx]

        out_state_grads = [exec_train.outputs[i].copyto(exec_train.outputs[i].context) \
                for i in out_state_idx]

        idx_rules = []
        for rule in self.rules:
            i_input = state_names.index(rule.input)
            i_output = output_names.index(rule.output)
            idx_rules.append(RecurrentRule(output=i_output, input=i_input, t=rule.t))

        updater = get_updater(optimizer)

        for epoch in range(begin_epoch, end_epoch):
            if eval_data is not None:
                eval_metric.reset()
                eval_data.reset()

                for eval_batch in eval_data:
                    seq_len = eval_batch.sequence_length
                    fwd_states = deque(maxlen=seq_len)

                    eval_metric.begin_sequence()
                    for t in range(seq_len):
                        self.load_data(eval_batch.data_at(t), data_arrays)

                        # copy states over
                        for rule in idx_rules:
                            if t >= rule.t:
                                fwd_states[t-rule.t][rule.output].copyto(state_arrays[rule.input])
                            else:
                                state_arrays[rule.input][:] = 0

                       # forward 1 step
                        exec_train.forward(is_train=False)

                        # save states
                        fwd_states.append([x.copyto(x.context) for x in exec_train.outputs])

                        pred_output = exec_train.outputs[:n_loss]
                        #if len(pred_output) == 1:
                        #    pred_output = pred_output[0]
                        eval_metric.update_at(eval_batch.label_at(t), pred_output, t)
                    eval_metric.end_sequence()

                name, value = eval_metric.get()
                print('Epoch[%d] Validation-%s=%f' % (epoch, name, value))

            data.reset()
            for batch in data:
                seq_len = batch.sequence_length
                fwd_states = []

                # forward pass through time
                for t in range(seq_len):
                    # assuming forward stage labels are not needed
                    self.load_data(batch.data_at(t), data_arrays)

                    # copy states over
                    for rule in idx_rules:
                        state_arrays[rule.input][:] = 0
                        if t >= rule.t:
                            fwd_states[t-rule.t][rule.output].copyto(state_arrays[rule.input])
                        else:
                            state_arrays[rule.input][:] = 0

                    # forward 1 step
                    exec_train.forward(is_train=True)

                    # save states
                    fwd_states.append([x.copyto(x.context) for x in exec_train.outputs])

                bwd_states = deque(maxlen=self.max_delay)

                # backward pass through time
                for grad in param_grads:
                    grad[:] = 0

                for t in reversed(range(seq_len)):
                    # load data and label
                    self.load_data(batch.data_at(t), data_arrays)
                    self.load_data(batch.label_at(t), label_arrays)

                    # need to run forward one more time, as the intermediate
                    # states for time index t during forward has been destroyed
                    for rule in idx_rules:
                        state_arrays[rule.input][:] = 0
                        if t >= rule.t:
                            fwd_states[t-rule.t][rule.output].copyto(state_arrays[rule.input])
                        else:
                            state_arrays[rule.input][:] = 0

                        exec_train.forward(is_train=True)

                    # save memory, the last fwd-state is no longer needed
                    fwd_states.pop()

                    # now we can run backward

                    # copy state grads over
                    for rule in idx_rules:
                        out_state_grads[rule.output-n_loss][:] = 0
                        if t + rule.t < seq_len:
                            bwd_states[-rule.t][rule.input].copyto(out_state_grads[rule.output-n_loss])
                        else:
                            out_state_grads[rule.output-n_loss][:] = 0

                    exec_train.backward(out_state_grads)

                    # save state gradients
                    bwd_states.append([x.copyto(x.context) for x in state_grads])

                bwd_states.clear()

                # update parameters
                for index, pair in enumerate(zip(param_arrays, param_grads)):
                    the_param, the_grad = pair
                    if the_grad is None:
                        continue
                    updater(index, the_grad, the_param)


    @staticmethod
    def load_data(src, dst):
        for d_src, d_dst in zip(src, dst):
            d_src.copyto(d_dst)


    @staticmethod
    def extract_states(sym, rules):
        outputs = sym.list_outputs()
        states  = [x.output for x in rules]
        sym_all = sym.get_internals()
        sym_grp = symbol.Group([sym_all[x] for x in outputs+states])
        return sym_grp





