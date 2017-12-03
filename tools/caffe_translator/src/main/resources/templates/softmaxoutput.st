<var> = mx.sym.SoftmaxOutput(data=<data>, label=<label>, name='<name>')
<var>_metric = mx.metric.CrossEntropy(output_names=['<name>_output'], label_names=['<label_name>'], name='<name>/metric')
train_metrics.add(<var>_metric)
