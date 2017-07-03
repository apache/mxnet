import mxnet as mx
from mxnet.foo import nn, rnn
from mxnet.foo.parameter import Parameter
import logging

class ChildSumLSTMCell(rnn.RecurrentCell):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer=None,
                 hs2h_bias_initializer=None,
                 hc2h_bias_initializer=None,
                 input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._reg_params = {}
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


    def __setattr__(self, name, value):
        """Registers parameters."""
        super(ChildSumLSTMCell, self).__setattr__(name, value)
        if isinstance(value, Parameter):
            assert name not in self._reg_params or \
                not isinstance(self._reg_params[name], Parameter), \
                "Overriding Parameter attribute %s is not allowed. " \
                "Please pass in Parameters by specifying `params` at " \
                "Layer construction instead."
            self._reg_params[name] = value

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

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
            params = {i: j.data(ctx) for i, j in self._reg_params.items()}
            return self.node_forward(F, F.expand_dims(inputs[tree.idx], axis=0), children_states, **params)

    def node_forward(self, F, inputs, children_states,
                     i2h_weight, hs2h_weight, hc2h_weight,
                     i2h_bias, hs2h_bias, hc2h_bias):
        name = self._curr_prefix

        logging.debug('inputs: ')
        logging.debug(inputs)
        logging.debug('children_states: ')
        logging.debug(children_states)
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4,
                               name='%si2h'%name) # {i, f, u, o} (N, 4*C)
        slice_i2h = F.split(i2h, num_outputs=4, name='%siuo_slice'%name) # (N, C)*4
        iuo_gates = F.concat(*[slice_i2h[i] for i in [0, 2, 3]], dim=1) # (N, C*3)
        if children_states:
            hs = F.add_n(*[state[0] for state in children_states], name='%shs'%name) # (N, C)
            logging.debug('hs: %s', str(hs.shape))
            hc = F.concat(*[F.expand_dims(state[0], axis=1) for state in children_states], dim=1,
                          name='%shc') # (N, K, C)
            logging.debug('hc: %s', str(hc.shape))
            cs = F.concat(*[F.expand_dims(state[1], axis=1) for state in children_states], dim=1,
                          name='%scs') # (N, K, C)
            logging.debug('cs: %s', str(cs.shape))
            f_act = slice_i2h[1] + hc2h_bias + F.dot(hc, hc2h_weight) # (N, K, C)
            logging.debug('f_act: %s', str(f_act.shape))
            logging.debug('iuo_gates: %s', str(iuo_gates.shape))
        else:
            hs = F.zeros_like(slice_i2h[0])

        hs2h = F.FullyConnected(data=hs, weight=hs2h_weight, bias=hs2h_bias,
                                num_hidden=self._hidden_size*3,
                                name='%shs2h'%name) # {i, u, o}
        logging.debug('hs2h: %s', str(hs2h.shape))
        iuo_gates = iuo_gates + hs2h

        slice_iuo = F.SliceChannel(iuo_gates, num_outputs=3,
                                   name='%sslice'%name) # (N, C)*3
        logging.debug('slice_iuo: %s', str([x.shape for x in slice_iuo]))

        in_gate = F.Activation(slice_iuo[0], act_type='sigmoid', name='%si'%name)
        logging.debug('in_gate: %s', str(in_gate.shape))
        logging.debug(in_gate)
        in_transform = F.Activation(slice_iuo[1], act_type='tanh', name='%sc'%name)
        logging.debug('in_transform: %s', str(in_transform.shape))
        logging.debug(in_transform)
        out_gate = F.Activation(slice_iuo[2], act_type='sigmoid', name='%so'%name)
        logging.debug('out_gate: %s', str(out_gate.shape))
        logging.debug(out_gate)

        next_c = in_gate * in_transform
        if children_states:
            forget_gates = F.Activation(f_act, act_type='sigmoid', name='%sf'%name) # 1:(N, C) 2:(N,K,C)
            next_c = F._internal._plus(F.sum(forget_gates * cs, axis=1), next_c,
                                       name='%sstate'%name)
            logging.debug(forget_gates)
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type='tanh'),
                                  name='%sout'%name)

        return next_h, [next_h, next_c]

# module for distance-angle similarity
class Similarity(nn.Layer):
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        with self.name_scope():
            self.wh = nn.Dense(sim_hidden_size, in_units=2*rnn_hidden_size, prefix='sim_embed_')
            self.wp = nn.Dense(num_classes, in_units=sim_hidden_size, prefix='sim_out_')

    def forward(self, F, lvec, rvec):
        mult_dist = F.broadcast_mul(lvec, rvec)
        abs_dist = F.abs(F.add(lvec,-rvec))
        vec_dist = F.concat(*[mult_dist, abs_dist],dim=1)
        out = F.log_softmax(self.wp(F.sigmoid(self.wh(vec_dist))))
        return out

# putting the whole model together
class SimilarityTreeLSTM(nn.Layer):
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
