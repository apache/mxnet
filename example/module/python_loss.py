import numpy as np
import mxnet as mx
import numba
import logging

class MulticlassHingeOutput(mx.mod.PythonModule):
    def __init__(self, name='mc-hinge', data_names=['data'],
                 label_names=['label'], logger=logging):
        super(MulticlassHingeOutput, self).__init__(data_names=data_names,
                                                    label_names=label_names,
                                                    output_names=[name + '_output'],
                                                    logger=logger)
        self._name = name
        assert len(data_names) == 1
        assert len(label_names) == 1

        self._scores = None
        self._labels = None
        self._scores_grad = None

    def _compute_output_shapes(self):
        """Compute the shapes of outputs. As a loss module with outputs, we simply
        output whatever we receive as inputs (i.e. the scores).
        """
        return [(self._name + '_output', self._data_shapes[0][1])]

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is `None`, which means `is_train` takes the value of `self.for_training`.
        """
        self._scores = data_batch.data[0]
        self._labels = data_batch.label[0]

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation. As a output loss module,
        we treat the inputs to this module as scores, and simply return them.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be `True`, because we do not use multiple contexts for computing.
        """
        assert merge_multi_context == True
        return [self._scores]

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert out_grads is None, 'MulticlassHingeOutput is a loss module, out_grads should be None'
        assert self.for_training

        self._scores_grad = mx.nd.array(mc_hinge_grad(self._scores.asnumpy(),
                                                      self._labels.asnumpy()))

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients to the inputs, computed in the previous backward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be `True` because we do not use multiple context for computation.
        """
        return [self._scores_grad]

@numba.jit
def mc_hinge_grad(scores, labels):
    n, _ = scores.shape
    grad = np.zeros_like(scores)

    for i in range(n):
        score = 1 + scores[i] - scores[i, labels[i]]
        score[labels[i]] = 0
        ind_pred = score.argmax()
        grad[i, labels[i]] -= 1
        grad[i, ind_pred] += 1

    return grad

if __name__ == '__main__':
    n_epoch = 2
    batch_size = 100
    contexts = [mx.context.cpu()]

    # build a MLP module
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)

    mlp = mx.mod.Module(fc3, context=contexts)
    loss = MulticlassHingeOutput()

    mod = mx.mod.SequentialModule().add(mlp).add(loss, take_labels=True, auto_wiring=True)

    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
    val_dataiter = mx.io.MNISTIter(
            image="data/t10k-images-idx3-ubyte",
            label="data/t10k-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=True, flat=True, silent=False)

    logging.basicConfig(level=logging.DEBUG)
    mod.fit(train_dataiter, eval_data=val_dataiter,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
            num_epoch=n_epoch)
