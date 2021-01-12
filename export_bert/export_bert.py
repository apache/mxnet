import os
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet.contrib import onnx as onnx_mxnet
import onnxruntime as rt
import onnx
import time
from onnxsim import simplify
from mxnet.test_utils import assert_almost_equal

parser = argparse.ArgumentParser(description="Tune/evaluate Bert model")
parser.add_argument("--layer", type=int, default=12,
                    help="Number of layers in BERT model (default: 12)")
parser.add_argument("--task", choices=["classification", "regression", "question_answering"],
                    default="classification",
                    help="specify the model type (default: classification)")
args = parser.parse_args()

ctx = mx.cpu(0)
use_pooler = False if args.task == "question_answering" else True
model_name='bert_12_768_12'
dataset='book_corpus_wiki_en_uncased'
bert, _ = nlp.model.get_model(
    name=model_name,
    ctx=ctx,
    dataset_name=dataset,
    pretrained=True,
    use_pooler=True,
    use_decoder=False,
    num_layers=3, # hardcode this as 3 layer since this is what the customer uses
    use_classifier=False,
    hparam_allow_override=True)
model = bert
# using pretrained weights, no need to initilize
# model.initialize(ctx=ctx)
model.hybridize(static_alloc=False)

batch = 128
seq_length = 32
inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32', ctx=ctx)
valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)
seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

model_dir = f'bert_model'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

prefix = '%s/mx_bert_layer%s' % (model_dir, args.layer)
model.export(prefix)

sym_file = "%s-symbol.json" % prefix
params_file = "%s-0000.params" % prefix
onnx_file = "%s.onnx" % prefix
input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
converted_model_path = onnx_mxnet.export_model(sym_file, params_file, input_shapes,
                                               np.float32, onnx_file, verbose=True)


def simplify_model(onnx_file, onnx_file_simp):
    model = onnx.load(onnx_file)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_file_simp)

onnx_file_simp = "%s_simp.onnx" % prefix
simplify_model(onnx_file, onnx_file_simp)


''' Benchmark Code

#rt.set_default_logger_severity(0)
sess_options = rt.SessionOptions()
#sess_options.enable_profiling = True
#sess_options.intra_op_num_threads=1
#sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
#sess_options.inter_op_num_threads = 2
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = rt.InferenceSession(onnx_file_simp, sess_options)

mx_ = []
on_ = []
for i in range(1):
    ctx = mx.cpu()
    mx.random.seed(int(time.time()*100000))
    inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
    token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32', ctx=ctx)
    valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)

    # mxnet
    t0 = time.time()
    seq_encoding, cls_encoding = model(inputs.copyto(mx.cpu(0)),
                                       token_types.copyto(mx.cpu(0)),
                                       valid_length.copyto(mx.cpu(0)))
    seq_encoding.wait_to_read()
    cls_encoding.wait_to_read()
    t1 = time.time()
    print('mx-nat:', t1-t0)
    mx_.append(t1-t0)
    print(seq_encoding, cls_encoding)
    print("-------------------------")

    # onnxruntime
    t0 = time.time()
    in_tensors = [inputs, token_types, valid_length]
    input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
    pred = sess.run(None, input_dict)
    t1 = time.time()
    print('onnxrt:', t1-t0)
    on_.append(t1-t0)
    print(pred[0], pred[1])

    # compare
    #assert_almost_equal(seq_encoding, pred[0])
    #assert_almost_equal(cls_encoding, pred[1])

print()
print()
print(rt.get_device())
print('mx avg:', sum(mx_)/len(mx_))
print('on avg:', sum(on_)/len(on_))
'''
