<iter_name> = mx.io.CaffeDataIter(
    prototxt =
<prototxt>,
    data_name='<data_name>',
    label_name='<label_name>',
<if(num_examples)>
    num_examples=<num_examples>,
<endif>
    flat = False
)
