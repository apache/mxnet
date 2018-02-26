# coding: utf-8
"""import function"""
import onnx
from .import_onnx import GraphProto

def import_model(model_file):
    """Imports the supplied ONNX model file into MXNet symbol and parameters.

    Parameters
    ----------
    model_file : ONNX model file name

    Returns
    -------
    sym : mx.symbol
        Compatible mxnet symbol

    params : dict of str to mx.ndarray
        Dict of converted parameters stored in mx.ndarray format
    """
    graph = GraphProto()

    # loads model file and returns ONNX protobuf object
    model_proto = onnx.load(model_file)
    sym, params = graph.from_onnx(model_proto.graph)
    return sym, params