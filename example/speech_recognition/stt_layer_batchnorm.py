import mxnet as mx


def batchnorm(net,
              gamma=None,
              beta=None,
              eps=0.001,
              momentum=0.9,
              fix_gamma=True,
              use_global_stats=False,
              output_mean_var=False,
              name=None):
    if gamma is not None and beta is not None:
        net = mx.sym.BatchNorm(data=net,
                               gamma=gamma,
                               beta=beta,
                               eps=eps,
                               momentum=momentum,
                               fix_gamma=fix_gamma,
                               use_global_stats=use_global_stats,
                               output_mean_var=output_mean_var
                               )
    else:
        net = mx.sym.BatchNorm(data=net,
                               eps=eps,
                               momentum=momentum,
                               fix_gamma=fix_gamma,
                               use_global_stats=use_global_stats,
                               output_mean_var=output_mean_var
                               )
    return net
