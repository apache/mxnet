import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

onnx_file = "/mx_bert_layer3_simp_1_16.onnx"
trt_file = '/mx_bert_layer3_simp_1_16.trt'

def build_engine(model_file, max_ws=512*1024*1024, fp16=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    explicit_batch = 1 << int (trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            #last_layer = network.get_layer(network.num_layers - 1)
            #network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine

engine = build_engine(onnx_file, fp16=True)

with open(trt_file, 'wb') as f:
    f.write(bytearray(engine.serialize()))
