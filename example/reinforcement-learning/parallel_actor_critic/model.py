from itertools import chain
import numpy as np
import scipy.signal
import mxnet as mx


class Agent(object):
    def __init__(self, input_size, act_space, config):
        super(Agent, self).__init__()
        self.input_size = input_size
        self.num_envs = config.num_envs
        self.ctx = config.ctx
        self.act_space = act_space
        self.config = config

        # Shared network.
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=config.hidden_size, no_bias=True)
        net = mx.sym.Activation(data=net, name='relu1', act_type="relu")

        # Policy network.
        policy_fc = mx.sym.FullyConnected(
            data=net, name='policy_fc', num_hidden=act_space, no_bias=True)
        policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy')
        policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
        log_policy = mx.sym.log(data=policy, name='log_policy')
        out_policy = mx.sym.BlockGrad(data=policy, name='out_policy')

        # Negative entropy.
        neg_entropy = policy * log_policy
        neg_entropy = mx.sym.MakeLoss(
            data=neg_entropy, grad_scale=config.entropy_wt, name='neg_entropy')

        # Value network.
        value = mx.sym.FullyConnected(data=net, name='value', num_hidden=1)

        self.sym = mx.sym.Group([log_policy, value, neg_entropy, out_policy])
        self.model = mx.mod.Module(self.sym, data_names=('data',),
                                   label_names=None)

        self.paralell_num = config.num_envs * config.t_max
        self.model.bind(
            data_shapes=[('data', (self.paralell_num, input_size))],
            label_shapes=None,
            grad_req="write")

        self.model.init_params(config.init_func)

        optimizer_params = {'learning_rate': config.learning_rate,
                            'rescale_grad': 1.0}
        if config.grad_clip:
            optimizer_params['clip_gradient'] = config.clip_magnitude

        self.model.init_optimizer(
            kvstore='local', optimizer=config.update_rule,
            optimizer_params=optimizer_params)

    def act(self, ps):
        us = np.random.uniform(size=ps.shape[0])[:, np.newaxis]
        as_ = (np.cumsum(ps, axis=1) > us).argmax(axis=1)
        return as_

    def train_step(self, env_xs, env_as, env_rs, env_vs):
        # NOTE(reed): Reshape to set the data shape.
        self.model.reshape([('data', (len(env_xs), self.input_size))])

        xs = mx.nd.array(env_xs, ctx=self.ctx)
        as_ = np.array(list(chain.from_iterable(env_as)))

        # Compute discounted rewards and advantages.
        advs = []
        gamma, lambda_ = self.config.gamma, self.config.lambda_
        for i in xrange(len(env_vs)):
            # Compute advantages using Generalized Advantage Estimation;
            # see eqn. (16) of [Schulman 2016].
            delta_t = (env_rs[i] + gamma*np.array(env_vs[i][1:]) -
                       np.array(env_vs[i][:-1]))
            advs.extend(self._discount(delta_t, gamma * lambda_))

        # Negative generalized advantage estimations.
        neg_advs_v = -np.asarray(advs)

        # NOTE(reed): Only keeping the grads for selected actions.
        neg_advs_np = np.zeros((len(advs), self.act_space), dtype=np.float32)
        neg_advs_np[np.arange(neg_advs_np.shape[0]), as_] = neg_advs_v
        neg_advs = mx.nd.array(neg_advs_np, ctx=self.ctx)

        # NOTE(reed): The grads of values is actually negative advantages.
        v_grads = mx.nd.array(self.config.vf_wt * neg_advs_v[:, np.newaxis],
                              ctx=self.ctx)

        data_batch = mx.io.DataBatch(data=[xs], label=None)
        self._forward_backward(data_batch=data_batch,
                               out_grads=[neg_advs, v_grads])

        self._update_params()

    def _discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def _forward_backward(self, data_batch, out_grads=None):
        self.model.forward(data_batch, is_train=True)
        self.model.backward(out_grads=out_grads)

    def _update_params(self):
        self.model.update()
        self.model._sync_params_from_devices()
