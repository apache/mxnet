# pylint: skip-file
import mxnet as mx
import numpy as np
import logging

class Monitor(object):
    def __init__(self, interval, level=logging.DEBUG, stat=None):
        self.interval = interval
        self.level = level
        if stat is None:
            def mean_abs(x):
                return np.fabs(x).mean()
            self.stat = mean_abs
        else:
            self.stat = stat

    def forward_end(self, i, internals):
        if i%self.interval == 0 and logging.getLogger().isEnabledFor(self.level):
            for key in sorted(internals.keys()):
                arr = internals[key]
                logging.log(self.level, 'iter:%d  param:%s\t\tstat(%s):%s'%(i, key, self.stat.__name__, str(self.stat(arr.asnumpy()))))

    def backward_end(self, i, weights, grads, metric=None):
        if i%self.interval == 0 and logging.getLogger().isEnabledFor(self.level):
            for key in sorted(grads.keys()):
                arr = grads[key]
                logging.log(self.level, 'iter:%d  param:%s\t\tstat(%s):%s\t\tgrad_stat:%s'%(i, key, self.stat.__name__, str(self.stat(weights[key].asnumpy())), str(self.stat(arr.asnumpy()))))
            if metric is not None:
                logging.info('Iter:%d metric:%f'%(i, metric.get()[1]))

class Solver(object):
    def __init__(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            self.optimizer = mx.optimizer.create(optimizer, **kwargs)
        else:
            self.optimizer = optimizer
        self.updater = mx.optimizer.get_updater(self.optimizer)
        self.monitor = None
        self.metric = None
        self.iter_end_callback = None
        self.iter_start_callback = None

    def set_metric(self, metric):
        self.metric = metric

    def set_monitor(self, monitor):
        self.monitor = monitor

    def set_iter_end_callback(self, callback):
        self.iter_end_callback = callback

    def set_iter_start_callback(self, callback):
        self.iter_start_callback = callback

    def solve(self, xpu, sym, args, args_grad, input_names,
              data_iter, begin_epoch, end_epoch, debug = False, args_lrmult=None):
        if args_lrmult is None:
            args_lrmult = {}

        data_iter.reset()
        input_dict = {key: mx.nd.empty(arr.shape, ctx=xpu) for key, arr in zip(input_names, data_iter.next())}
        batch_size = input_dict.values()[0].shape[0]
        self.optimizer.rescale_grad = 1.0/batch_size
        args = dict(args, **input_dict)

        output_names = sym.list_outputs()
        if debug:
            sym = sym.get_internals()
            blob_names = sym.list_outputs()
            sym_group = []
            for i in range(len(blob_names)):
                if blob_names[i] not in args:
                    x = sym[i]
                    if blob_names[i] not in output_names:
                        x = mx.symbol.BlockGrad(x, name=blob_names[i])
                    sym_group.append(x)
            sym = mx.symbol.Group(sym_group)
        exe = sym.bind(xpu, args=args, args_grad=args_grad)

        update_dict = {name: args_grad[name] for name, nd in zip(sym.list_arguments(), exe.grad_arrays) if nd}

        output_dict = {}
        output_buff = {}
        internal_dict = {}
        for key, arr in zip(sym.list_outputs(), exe.outputs):
            if key in output_names:
                output_dict[key] = arr
                output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
            else:
                internal_dict[key] = arr

        data_iter.reset()
        for i in range(begin_epoch, end_epoch):
            if self.iter_start_callback is not None:
                self.iter_start_callback(i)
            try:
                data_list = data_iter.next()
            except:
                data_iter.reset()
            for data, key in zip(data_list, input_names):
                data.copyto(input_dict[key])
            exe.forward(is_train=True)
            if self.monitor is not None:
                self.monitor.forward_end(i, internal_dict)
            for key in output_dict:
                output_dict[key].copyto(output_buff[key])

            exe.backward()
            self.optimizer.begin_epoch(i)
            for key, arr in update_dict.items():
                self.updater(key, arr, args[key], args_lrmult.get(key, 1.0))

            exe.outputs[0].wait_to_read()
            if self.metric is not None:
                self.metric.update(input_dict[input_names[-1]].asnumpy(),
                                   output_buff[output_names[0]].asnumpy())

            if self.monitor is not None:
                self.monitor.backward_end(i, args, update_dict, self.metric)

            if self.iter_end_callback is not None:
                self.iter_end_callback(i) 





        

