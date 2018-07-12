from test_tensorrt_lenet5 import test_tensorrt_inference
s, ex = test_tensorrt_inference()
optimized_s = ex.optimized_symbol
print(optimized_s.get_internals())
