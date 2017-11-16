import mxnet as mx


class Config(object):
    def __init__(self, args):
        # Default training settings
        self.ctx = mx.gpu(0) if args.gpu else mx.cpu()
        self.init_func = mx.init.Xavier(rnd_type='uniform', factor_type="in",
                                        magnitude=1)
        self.learning_rate = 1e-3
        self.update_rule = "adam"
        self.grad_clip = True
        self.clip_magnitude = 40

        # Default model settings
        self.hidden_size = 200
        self.gamma = 0.99
        self.lambda_ = 1.0
        self.vf_wt = 0.5        # Weight of value function term in the loss
        self.entropy_wt = 0.01  # Weight of entropy term in the loss

        self.num_envs = 16
        self.t_max = 50

        # Override defaults with values from `args`.
        for arg in self.__dict__:
            if arg in args.__dict__:
                self.__setattr__(arg, args.__dict__[arg])
