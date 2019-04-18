import mxnet as mx

#sym = mx.sym.load("resnet50_v1-symbol.json")
data = mx.sym.var("data")
data2 = mx.sym.var("data2")
data3 = mx.sym.var("data3")
x = mx.sym.exp(data)
x2 = mx.sym.sin(data)
x3 = mx.sym.cos(data)
sym = x + x2 + x3
result = mx.sym.add_n(sym, data2, data3)
#x = mx.viz.plot_network(result)
casted_result = mx.contrib.amp._convert_symbol(result, target_dtype="float16",
                                               target_dtype_ops=["sin", "cos", "exp"], fp32_ops=["elemwise_add"],
                                               widest_dtype_ops=["add_n"], conditional_fp32_ops=None)
#y = mx.viz.plot_network(casted_result, use_op_names=True)
#y.render('test-output/round-table.gv', view=False)
