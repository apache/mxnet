<if(fix_beta)>
<var>_beta = mx.sym.BlockGrad(mx.sym.Variable("<name>_beta", init=mx.init.Constant(0)))
<endif>
<var> = mx.symbol.BatchNorm(data=<data>,
<if(fix_beta)>
    beta=<var>_beta,
<endif>
<if(fix_gamma)>
    fix_gamma=True,
<endif>
<if(use_global_stats)>
    use_global_stats=True,
<endif>
    name='<name>')
