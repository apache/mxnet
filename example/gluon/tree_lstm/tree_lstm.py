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
from mxnet.gluon import Block, nn
from mxnet.gluon.parameter import Parameter

class ChildSumLSTMCell(Block):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 hs2h_bias_initializer='zeros',
                 hc2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.i2h_weight = self.params.get('i2h_weight', shape=(4*hidden_size, input_size),
                                              init=i2h_weight_initializer)
            self.hs2h_weight = self.params.get('hs2h_weight', shape=(3*hidden_size, hidden_size),
                                               init=hs2h_weight_initializer)
            self.hc2h_weight = self.params.get('hc2h_weight', shape=(hidden_size, hidden_size),
                                               init=hc2h_weight_initializer)
            self.i2h_bias = self.params.get('i2h_bias', shape=(4*hidden_size,),
                                            init=i2h_bias_initializer)
            self.hs2h_bias = self.params.get('hs2h_bias', shape=(3*hidden_size,),
                                             init=hs2h_bias_initializer)
            self.hc2h_bias = self.params.get('hc2h_bias', shape=(hidden_size,),
                                             init=hc2h_bias_initializer)

    def _alias(self):
        return 'childsum_lstm'

    def forward(self, F, inputs, tree):
        children_outputs = [self.forward(F, inputs, child)
                            for child in tree.children]
        if children_outputs:
            _, children_states = zip(*children_outputs) # unzip
        else:
            children_states = None

        with inputs.context as ctx:
            return self.node_forward(F, F.expand_dims(inputs[tree.idx], axis=0), children_states,
                                     self.i2h_weight.data(ctx),
                                     self.hs2h_weight.data(ctx),
                                     self.hc2h_weight.data(ctx),
                                     self.i2h_bias.data(ctx),
                                     self.hs2h_bias.data(ctx),
                                     self.hc2h_bias.data(ctx))

    def node_forward(self, F, inputs, children_states,
                     i2h_weight, hs2h_weight, hc2h_weight,
                     i2h_bias, hs2h_bias, hc2h_bias):
        name = '{0}{1}_'.format(self.prefix, self._alias)
        # notation: N for batch size, C for hidden state dimensions, K for number of children.

        # FC for i, f, u, o gates (N, 4*C), from input to hidden
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4,
                               name='%si2h'%name)
        i2h_slices = F.split(i2h, num_outputs=4, name='%siuo_slice'%name) # (N, C)*4
        i2h_iuo = F.concat(*[i2h_slices[i] for i in [0, 2, 3]], dim=1) # (N, C*3)
        if children_states:
            # sum of children states
            hs = F.add_n(*[state[0] for state in children_states], name='%shs'%name) # (N, C)
            # concatenation of children hidden states
            hc = F.concat(*[F.expand_dims(state[0], axis=1) for state in children_states], dim=1,
                          name='%shc') # (N, K, C)
            # concatenation of children cell states
            cs = F.concat(*[F.expand_dims(state[1], axis=1) for state in children_states], dim=1,
                          name='%scs') # (N, K, C)

            # calculate activation for forget gate. addition in f_act is done with broadcast
            i2h_f_slice = i2h_slices[1]
            f_act = i2h_f_slice + hc2h_bias + F.dot(hc, hc2h_weight) # (N, K, C)
            forget_gates = F.Activation(f_act, act_type='sigmoid', name='%sf'%name) # (N, K, C)
        else:
            # for leaf nodes, summation of children hidden states are zeros.
            hs = F.zeros_like(i2h_slices[0])

        # FC for i, u, o gates, from summation of children states to hidden state
        hs2h_iuo = F.FullyConnected(data=hs, weight=hs2h_weight, bias=hs2h_bias,
                                    num_hidden=self._hidden_size*3,
                                    name='%shs2h'%name)
        i2h_iuo = i2h_iuo + hs2h_iuo

        iuo_act_slices = F.SliceChannel(i2h_iuo, num_outputs=3,
                                        name='%sslice'%name) # (N, C)*3
        i_act, u_act, o_act = iuo_act_slices[0], iuo_act_slices[1], iuo_act_slices[2] # (N, C) each

        # calculate gate outputs
        in_gate = F.Activation(i_act, act_type='sigmoid', name='%si'%name)
        in_transform = F.Activation(u_act, act_type='tanh', name='%sc'%name)
        out_gate = F.Activation(o_act, act_type='sigmoid', name='%so'%name)

        # calculate cell state and hidden state
        next_c = in_gate * in_transform
        if children_states:
            next_c = F._internal._plus(F.sum(forget_gates * cs, axis=1), next_c,
                                       name='%sstate'%name)
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type='tanh'),
                                  name='%sout'%name)

        return next_h, [next_h, next_c]

# module for distance-angle similarity
class Similarity(nn.Block):
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        with self.name_scope():
            self.wh = nn.Dense(sim_hidden_size, in_units=2*rnn_hidden_size, prefix='sim_embed_')
            self.wp = nn.Dense(num_classes, in_units=sim_hidden_size, prefix='sim_out_')

    def forward(self, F, lvec, rvec):
        # lvec and rvec will be tree_lstm cell states at roots
        mult_dist = F.broadcast_mul(lvec, rvec)
        abs_dist = F.abs(F.add(lvec,-rvec))
        vec_dist = F.concat(*[mult_dist, abs_dist],dim=1)
        out = F.log_softmax(self.wp(F.sigmoid(self.wh(vec_dist))))
        return out

# putting the whole model together
class SimilarityTreeLSTM(nn.Block):
    def __init__(self, sim_hidden_size, rnn_hidden_size, embed_in_size, embed_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.embed = nn.Embedding(embed_in_size, embed_dim, prefix='word_embed_')
            self.childsumtreelstm = ChildSumLSTMCell(rnn_hidden_size, input_size=embed_dim)
            self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)

    def forward(self, F, l_inputs, r_inputs, l_tree, r_tree):
        l_inputs = self.embed(l_inputs)
        r_inputs = self.embed(r_inputs)
        lstate = self.childsumtreelstm(F, l_inputs, l_tree)[1][1]
        rstate = self.childsumtreelstm(F, r_inputs, r_tree)[1][1]
        output = self.similarity(F, lstate, rstate)
        return output
