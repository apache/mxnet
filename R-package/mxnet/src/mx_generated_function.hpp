
#ifndef mx_generated_function_hpp
#define mx_generated_function_hpp

#include "ndarray.hpp"
#include "mxnet.h"

namespace Rcpp {

class MxFunction : public CppFunction {
    public:
        MxFunction(FunctionHandle handle, const char * docstring = 0):handle(handle),
            CppFunction(docstring) {
            int ret = MXFuncGetInfo(handle, &name, &desc, &num_args, 
                                    &arg_names, &arg_types, &arg_descs);    
            // remove the '_'
            if (name[0] == '_')
                name++;
        }

        SEXP operator()(SEXP * args) {
            BEGIN_RCPP
            NDArrayHandle out;
            int ret = MXNDArrayCreateNone(&out);
            NDArray res(out);
            NDArrayHandle * use_vars = (NDArrayHandle *)malloc(num_args * sizeof(NDArrayHandle));
            for (int i = 0; i < num_args; i++) {
                Rcpp::XPtr<NDArray> ptr(args[i]);
                use_vars[i] = (*ptr.get()).handle;
            }

            MXFuncInvoke(handle, use_vars, NULL, &res.handle);
            return Rcpp::XPtr<NDArray>(new NDArray(res));
            END_RCPP
        }

        inline int nargs() { return num_args; }
        inline bool is_void() { return false; }
        inline void signature(std::string& s, const char* name) { Rcpp::signature<void_type>(s, name); }

        inline const char * get_name() {return name; };

        inline DL_FUNC get_function_ptr() { return (DL_FUNC)NULL; }
    private:
        FunctionHandle handle;
        const char * name;
        const char * desc;
        mx_uint num_args;
        const char ** arg_names;
        const char ** arg_types;
        const char ** arg_descs;

};

}

#endif
