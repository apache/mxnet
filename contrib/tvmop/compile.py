import tvm

import os
import argparse
from tvmop.opdef import __OP_DEF__

def get_target(device):
    if device == "cpu":
        # TODO: add llvm options
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-i", action="store", required=True, dest="input_path",
                        help="Input path where operators are defined")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    arguments = parser.parse_args()

    func_list_llvm = []
    func_list_cuda = []

    for operator_def in __OP_DEF__:
        for sch, args in operator_def.invoke_all():
            func_list = func_list_llvm if operator_def.target == "cpu" else func_list_cuda
            func_lower = tvm.lower(sch, args,
                                   name=operator_def.get_op_name(args),
                                   binds=operator_def.get_binds(args))
            func_list.append(func_lower)

    lowered_funcs = {get_target("cpu") : func_list_llvm}
    if len(func_list_cuda) > 0:
        lowered_funcs[get_target("cuda")] = func_list_cuda
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    func_binary.export_library(arguments.target_path)
