import tvm
import vector_add

funcs_list = []
for func in dir(vector_add):
    if func.startswith('defop_'):
        f = getattr(vector_add, func)
        f_name = func.split('defop_')[1]
        print("Compile", f_name, "...")
        sch, args = f()
        func_lower = tvm.lower(sch, args, name=f_name)
        funcs_list.append(func_lower)

lowered_funcs = {"llvm" : funcs_list}
func_binary = tvm.build(lowered_funcs, name="tvmop")

from tvm.contrib import cc
temp = "/home/ubuntu/incubator-mxnet/lib"
func_binary.save(temp + "/libtvmop.o")
cc.create_shared(temp + "/libtvmop.so", [temp + "/libtvmop.o"])
