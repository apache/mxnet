<var> = mx.symbol.Pooling(data=<data>,
    pool_type='<type>',
<if(global_pool)>
    global_pool=<global_pool>,
<endif>
<if(kernel_h)>
    kernel=(<kernel_h>,<kernel_w>),
<endif>
<if(stride_h)>
    stride=(<stride_h>,<stride_w>),
<endif>
<if(pad_h)>
    pad=(<pad_h>,<pad_w>),
<endif>
    pooling_convention='full',
    name='<name>')
