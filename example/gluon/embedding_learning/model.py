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


from mxnet import gluon
from mxnet.gluon import nn, Block, HybridBlock
import numpy as np

class L2Normalization(HybridBlock):
    r"""Applies L2 Normalization to input.

    Parameters
    ----------
    mode : str
        Mode of normalization.
        See :func:`~mxnet.ndarray.L2Normalization` for available choices.

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, mode, **kwargs):
        self._mode = mode
        super(L2Normalization, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.L2Normalization(x, mode=self._mode, name='l2_norm')

    def __repr__(self):
        s = '{name}({_mode})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


def get_distance(F, x):
    """Helper function for margin-based loss. Return a distance matrix given a matrix."""
    n = x.shape[0]

    square = F.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * F.dot(x, x.transpose()))

    # Adding identity to make sqrt work.
    return F.sqrt(distance_square + F.array(np.identity(n)))

class DistanceWeightedSampling(HybridBlock):
    r"""Distance weighted sampling. See "sampling matters in deep embedding learning"
    paper for details.

    Parameters
    ----------
    batch_k : int
        Number of images per class.

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """
    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
        self.batch_k = batch_k
        self.cutoff = cutoff

        # We sample only from negatives that induce a non-zero loss.
        # These are negatives with a distance < nonzero_loss_cutoff.
        # With a margin-based loss, nonzero_loss_cutoff == margin + beta.
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        super(DistanceWeightedSampling, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        k = self.batch_k
        n, d = x.shape

        distance = get_distance(F, x)
        # Cut off to avoid high variance.
        distance = F.maximum(distance, self.cutoff)

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(d)) * F.log(distance)
                       - (float(d - 3) / 2) * F.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = F.exp(log_weights - F.max(log_weights))

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = np.ones(weights.shape)
        for i in range(0, n, k):
            mask[i:i+k, i:i+k] = 0

        weights = weights * F.array(mask) * (distance < self.nonzero_loss_cutoff)
        weights = weights / F.sum(weights, axis=1, keepdims=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.asnumpy()
        for i in range(n):
            block_idx = i // k

            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1) * k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return a_indices, x[a_indices], x[p_indices], x[n_indices], x

    def __repr__(self):
        s = '{name}({batch_k})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class MarginNet(Block):
    r"""Embedding network with distance weighted sampling.
    It takes a base CNN and adds an embedding layer and a
    sampling layer at the end.

    Parameters
    ----------
    base_net : Block
        Base network.
    emb_dim : int
        Dimensionality of the embedding.
    batch_k : int
        Number of images per class in a batch. Used in sampling.

    Inputs:
        - **data**: input tensor with shape (batch_size, channels, width, height).
        Here we assume the consecutive batch_k images are of the same class.
        For example, if batch_k = 5, the first 5 images belong to the same class,
        6th-10th images belong to another class, etc.

    Outputs:
        - The output of DistanceWeightedSampling.
    """
    def __init__(self, base_net, emb_dim, batch_k, **kwargs):
        super(MarginNet, self).__init__(**kwargs)
        with self.name_scope():
            self.base_net = base_net
            self.dense = nn.Dense(emb_dim)
            self.normalize = L2Normalization(mode='instance')
            self.sampled = DistanceWeightedSampling(batch_k=batch_k)

    def forward(self, x):
        z = self.base_net(x)
        z = self.dense(z)
        z = self.normalize(z)
        z = self.sampled(z)
        return z


class MarginLoss(gluon.loss.Loss):
    r"""Margin based loss.

    Parameters
    ----------
    margin : float
        Margin between positive and negative pairs.
    nu : float
        Regularization parameter for beta.

    Inputs:
        - anchors: sampled anchor embeddings.
        - positives: sampled positive embeddings.
        - negatives: sampled negative embeddings.
        - beta_in: class-specific betas.
        - a_indices: indices of anchors. Used to get class-specific beta.

    Outputs:
        - Loss.
    """
    def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(MarginLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self._nu = nu

    def hybrid_forward(self, F, anchors, positives, negatives, beta_in, a_indices=None):
        if a_indices is not None:
            # Jointly train class-specific beta.
            beta = beta_in.data()[a_indices]
            beta_reg_loss = F.sum(beta) * self._nu
        else:
            # Use a constant beta.
            beta = beta_in
            beta_reg_loss = 0.0

        d_ap = F.sqrt(F.sum(F.square(positives - anchors), axis=1) + 1e-8)
        d_an = F.sqrt(F.sum(F.square(negatives - anchors), axis=1) + 1e-8)

        pos_loss = F.maximum(d_ap - beta + self._margin, 0.0)
        neg_loss = F.maximum(beta - d_an + self._margin, 0.0)

        pair_cnt = float(F.sum((pos_loss > 0.0) + (neg_loss > 0.0)).asscalar())

        # Normalize based on the number of pairs.
        loss = (F.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt
        return gluon.loss._apply_weighting(F, loss, self._weight, None)
