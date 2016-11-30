import os, sys
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
try:
    import mxnet as mx
except ImportError:
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "../../../python"))
    import mxnet as mx
