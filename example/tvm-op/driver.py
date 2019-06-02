import tvm
import vector_add

func_list_llvm = []
func_list_cuda = []
for func in dir(vector_add):
    if func.startswith('defop_'):
        f = getattr(vector_add, func)
        f_name = func.split('defop_')[1]
        if f_name.startswith("cuda_"):
            func_list = func_list_cuda
            print("Compile", f_name, "... target = cuda")
        else:
            func_list = func_list_llvm
            print("Compile", f_name, "... target = llvm")

        sch, args = f()
        func_lower = tvm.lower(sch, args, name=f_name)
        func_list.append(func_lower)

lowered_funcs = {"llvm" : func_list_llvm}
if len(func_list_cuda) > 0:
    lowered_funcs["cuda"] = func_list_cuda
func_binary = tvm.build(lowered_funcs, name="tvmop")

temp = "/home/ubuntu/mxnet/lib"
func_binary.export_library(temp + "/libtvmop.so")
