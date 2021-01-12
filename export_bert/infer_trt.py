import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import onnxruntime as rt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

trt_file = '/mx_bert_layer3_simp_1_16.trt'
onnx_file = '/mx_bert_layer3_simp_1_16.onnx'


with open(trt_file, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)

bert_context = engine.create_execution_context()


batch = 1
seq_length = 16
inputs = np.random.randint(0, 30522, size=(batch, seq_length)).astype('float32')
token_types = np.random.randint(0, 2, size=(batch, seq_length)).astype('float32')
valid_length = np.array([seq_length] * batch).astype('float32')
print(inputs.shape, token_types.shape, valid_length.shape)

out1 = np.zeros((batch, seq_length, 768)).astype('float32')
out2 = np.zeros((batch, 768)).astype('float32')
print(out1.shape, out2.shape)





d_inputs = cuda.mem_alloc(inputs.nbytes)
d_token_types = cuda.mem_alloc(token_types.nbytes)
d_valid_length = cuda.mem_alloc(valid_length.nbytes)

d_out1 = cuda.mem_alloc(out1.nbytes)
d_out2 = cuda.mem_alloc(out2.nbytes)

bindings = [int(d_inputs), int(d_token_types), int(d_valid_length), int(d_out1), int(d_out2)]



stream = cuda.Stream()
cuda.memcpy_htod_async(d_inputs, inputs, stream)
cuda.memcpy_htod_async(d_token_types, token_types, stream)
cuda.memcpy_htod_async(d_valid_length, valid_length, stream)

bert_context.execute_async(1, bindings, stream.handle, None)


cuda.memcpy_dtoh_async(out1, d_out1, stream)
cuda.memcpy_dtoh_async(out2, d_out2, stream)
stream.synchronize()

print('trt')
print(out1)
print('------------')
print(out2)

print()
print('~~~~~~~~~~~~')
print()


rt.set_default_logger_severity(4)
sess_options = rt.SessionOptions()
#sess_options.enable_profiling = True
#sess_options.intra_op_num_threads=1
#sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
#sess_options.inter_op_num_threads = 2
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = rt.InferenceSession(onnx_file, sess_options)

in_tensors = [inputs, token_types, valid_length]
input_dict = dict((sess.get_inputs()[i].name, in_tensors[i]) for i in range(len(in_tensors)))

print('onnx runtime')
pred = sess.run(None, input_dict)
print(pred[0])
print('------------')
print(pred[1])
