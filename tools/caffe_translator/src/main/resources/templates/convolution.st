<var> = mx.sym.Convolution(data=<data>,
    <if(weight)>weight=<weight>,<endif>
    <if(bias)>bias=<bias>,<endif>
    kernel=(<kernel_h>,<kernel_w>),
    stride=(<stride_h>,<stride_w>),
    pad=(<pad_h>,<pad_w>),
    num_filter=<num_filter>,
    <if(no_bias)>no_bias=True,<endif>
    name='<name>')
