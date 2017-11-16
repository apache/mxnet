from utils import define_policy
import mxnet as mx


class Policy(object):
    """
    Base class of policy.
    """

    def __init__(self, env_spec):

        self.env_spec = env_spec

    def get_actions(self, obs):

        raise NotImplementedError

    @property
    def observation_space(self):

        return self.env_spec.observation_space

    @property
    def action_space(self):

        return self.env_spec.action_space


class DeterministicMLPPolicy(Policy):
    """
    Deterministic Multi-Layer Perceptron Policy used
    for deterministic policy training.
    """

    def __init__(
        self,
        env_spec):

        super(DeterministicMLPPolicy, self).__init__(env_spec)

        self.obs = mx.symbol.Variable("obs")
        self.act = define_policy(
            self.obs, 
            self.env_spec.action_space.flat_dim)

    def get_output_symbol(self):

        return self.act

    def get_loss_symbols(self):

        return {"obs": self.obs,
                "act": self.act}

    def define_loss(self, loss_exp):
        """
        Define loss of the policy. No need to do so here.
        """

        raise NotImplementedError

    def define_exe(self, ctx, init, updater, input_shapes=None, args=None, 
                    grad_req=None):

        # define an executor, initializer and updater for batch version
        self.exe = self.act.simple_bind(ctx=ctx, **input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict

        for name, arr in self.arg_dict.items():
            if name not in input_shapes:
                init(name, arr)
                
        self.updater = updater

        # define an executor for sampled single observation
        # note the parameters are shared
        new_input_shapes = {"obs": (1, input_shapes["obs"][1])}
        self.exe_one = self.exe.reshape(**new_input_shapes)
        self.arg_dict_one = self.exe_one.arg_dict

    def update_params(self, grad_from_top):

        # policy accepts the gradient from the Value network
        self.exe.forward(is_train=True)
        self.exe.backward([grad_from_top])

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_actions(self, obs):

        # batch version
        self.arg_dict["obs"][:] = obs
        self.exe.forward(is_train=False)

        return self.exe.outputs[0].asnumpy()

    def get_action(self, obs):

        # single observation version
        self.arg_dict_one["obs"][:] = obs
        self.exe_one.forward(is_train=False)

        return self.exe_one.outputs[0].asnumpy()





        