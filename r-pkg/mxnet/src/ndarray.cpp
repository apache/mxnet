
#include "ndarray.hpp"
#include "mx_generated_function.hpp"

void NDArray::load(const std::string & filename) {

}

void NDArray::save(const std::string & filename) {

}

RCPP_MODULE(mod_ndarray) {
    using namespace Rcpp;
    class_<NDArray>("NDArray")
    .method("load", &NDArray::load)
    .method("save", &NDArray::save)
    ;

    mx_uint out_size;
    FunctionHandle * out_array;
    int ret = MXListFunctions(&out_size, &out_array);

    for (mx_uint i = 0; i < out_size; i++) {
        MxFunction fun(out_array[i]);    
        _rcpp_module_mod_ndarray.Add(fun.get_name(), &fun);
    }

}



