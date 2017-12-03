<var> = mx.sym.Deconvolution(data=<data>,
    <if(use_weight)>weight=weight,<endif>
    <if(use_bias)>bias=bias,<endif>
    kernel=(<kernel_h>,<kernel_w>),
    stride=(<stride_h>,<stride_w>),
    pad=(<pad_h>,<pad_w>),
    num_filter=<num_filter>,
    num_group=<group>,
    <if(no_bias)>no_bias=True,<endif>
    name='<name>')
