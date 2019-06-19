import tvm

import os
import argparse
import pkgutil

def get_target(device):
    if device == "cpu":
        # TODO: add llvm options
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


def get_operator_def(path):
    packages = [pkgname for importer, pkgname, ispkg in
                pkgutil.iter_modules([path]) if ispkg]
    packages = [path + os.sep + package for package in packages]
    operators = {}
    for importer, modname, ispkg in pkgutil.iter_modules(packages):
        if ispkg:
            continue
        module = importer.find_module(modname).load_module(modname)
        for func in dir(module):
            if func.startswith("defop_"):
                assert len(func) > len("defop_"), "Invalid function name " + func
                f_name = func.split('defop_')[1]
                assert f_name not in operators, "Duplicated definition " + f_name
                f = getattr(module, func)
                operators[f_name] = f
    assert len(operators) > 0, "Cannot find operator definition in " + path
    return operators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-i", action="store", required=True, dest="input_path",
                        help="Input path where operators are defined")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    arguments = parser.parse_args()

    operators = get_operator_def(arguments.input_path)

    func_list_llvm = []
    func_list_cuda = []
    for operator, func in operators.items():
        func_list = func_list_cuda if operator.startswith("cuda_") else func_list_llvm
        sch, args = func()

        binds = {}
        new_args = []
        for arg in args:
            if isinstance(arg, tuple):
                arg, buf = arg
                binds[arg] = buf
            new_args.append(arg)

        func_lower = tvm.lower(sch, new_args, name=operator, binds=binds)
        func_list.append(func_lower)

    lowered_funcs = {get_target("cpu") : func_list_llvm}
    if len(func_list_cuda) > 0:
        lowered_funcs[get_target("cuda")] = func_list_cuda
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    func_binary.export_library(arguments.target_path)
