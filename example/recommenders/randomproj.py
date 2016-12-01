"""Random projection layers in MXNet as custom python ops.
Currently slow and memory-inefficient, but functional.
"""
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"

import numpy as np
import mxnet as mx


# ref: http://mxnet.io/how_to/new_op.html

class RandomBagOfWordsProjection(mx.operator.CustomOp):
    """Random projection layer for sparse bag-of-words (n-hot) inputs.
    In the sparse input, only the indices are supplied, because all the
    values are understood to be exactly 1.0.

    See also RandomProjection for values other than 1.0.
    """

    def __init__(self, vocab_size, output_dim, random_seed=54321):
        # need_top_grad=True means this is not a loss layer
        super(RandomBagOfWordsProjection, self).__init__()
        self._vocab = vocab_size
        self._proj_dim = output_dim
        #NOTE: This naive implementation is slow and uses lots of memory.
        # Should use something smarter to not instantiate this matrix.
        rs = np.random.RandomState(seed=random_seed)
        self.W = self.random_unit_vecs(self._vocab, self._proj_dim, rs)

    def random_unit_vecs(self, num_vecs, num_dims, rs):
        W = rs.normal(size=(num_vecs, num_dims))
        Wlen = np.linalg.norm(W, axis=1)
        W_unit = W / Wlen[:,None]
        return W_unit

    def _get_mask(self, idx, in_data):
        """Returns the mask by which to multiply the parts of the embedding layer.
        In this version, we have no weights to apply.  
        """
        mask = idx >= 0  # bool False for -1 values that should be removed. shape=(b,mnz)
        mask = np.expand_dims(mask,2) # shape = (b,mnz,1)
        mask = np.repeat(mask, self._proj_dim, axis=2) # shape = (b,mnz,d)
        return mask

    def forward(self, is_train, req, in_data, out_data, aux):
        #Note: see this run in notebooks/howto-numpy-random-proj.ipynb
        # Notation for shapes: b = batch_size, mnz = max_nonzero, d = proj_dim
        idx = in_data[0].asnumpy().astype('int32') # shape=(b,mnz)

        wd = self.W[idx]  # shape= (b,mnz,d)
        mask = self._get_mask(idx, in_data)
        wd = np.multiply(wd,mask)  # shape=(b,mnz,d), but zero'd out non-masked
        y = np.sum(wd,axis=1)  # shape=(b,d)
        mxy = mx.nd.array(y)  #NOTE: this hangs if the environment variables aren't set correctly
        # See https://github.com/dmlc/mxnet/issues/3813
        self.assign(out_data[0], req[0], mxy)


@mx.operator.register("SparseBOWProj")
class RandomBagOfWordsProjectionProp(mx.operator.CustomOpProp):
    def __init__(self, vocab_size, output_dim):
        # need_top_grad=True means this is not a loss layer
        super(RandomBagOfWordsProjectionProp, self).__init__(need_top_grad=True)
        self._kwargs = {
            'vocab_size': int(vocab_size),
            'output_dim': int(output_dim),
        }

    def list_arguments(self):
        return ['indexes']

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return RandomBagOfWordsProjection(**self._kwargs)

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        output_shape = (batch_size, self._kwargs['output_dim'])
        return in_shape, [output_shape], []


class SparseRandomProjection(RandomBagOfWordsProjection):
    """Random projection of sparse input vector.
    Takes an sparse input layer, effectively in coordinate (COO) format,
    where the row number is implicit, because it's the minibatch record.

    See the simpler version RandomBagOfWordsProjection if all values are 1.0.
    """

    def _get_mask(self, idx, in_data):
        """Returns the mask by which to multiply the parts of the embedding layer.
        In this version, we apply the weights.
        """
        val = in_data[1].asnumpy()  # shape=(b,mnz)
        mask = idx >= 0  # bool False for -1 values that should be removed. shape=(b,mnz)
        mask = np.multiply(mask,val)  # All (b,mnz)
        mask = np.expand_dims(mask,2) # shape = (b,mnz,1)
        mask = np.repeat(mask, self._proj_dim, axis=2) # shape = (b,mnz,d)
        return mask
        

@mx.operator.register("SparseRandomProjection")
class SparseRandomProjectionProp(RandomBagOfWordsProjectionProp):

    def list_arguments(self):
        return ['indexes', 'values']

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return SparseRandomProjection(**self._kwargs)

    def infer_shape(self, in_shape):
        # check that indexes and values are the same shape.
        if in_shape[0] != in_shape[1]:
            raise ValueError("Input shapes differ. indexes:%s. values:%s. must be same"
                    % (str(in_shape[0]),str(in_shape[1])))
        return super(SparseRandomProjectionProp,self).infer_shape(in_shape)


if __name__ == "__main__":
    print("Simple test of proj layer")
    data = mx.symbol.Variable('data')
    vals = mx.symbol.Variable('vals')
    net = mx.symbol.Custom(indexes=data, values=vals, name='rproj', 
            op_type='SparseRandomProjection', 
            vocab_size=999, output_dim=29)
    d = mx.nd.zeros(shape=(3,100))
    v = mx.nd.ones(shape=(3,100))
    e = net.bind(ctx=mx.cpu(), args={'data':d, 'vals':v})
    e.forward()
    print(e.outputs[0].asnumpy())
    print("Done with proj layer test")

