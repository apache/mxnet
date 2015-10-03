
#include "ndarray.hpp"
#include "mx_generated_function.hpp"

void NDArray::load(const std::string & filename) {

}

void NDArray::save(const std::string & filename) {

}

RCPP_MODULE(NDArray) {
    using namespace Rcpp;
    class_<NDArray>("NDArray")
    .method("load", &NDArray::load)
    .method("save", &NDArray::save)
    ;

    mx_uint out_size;
    FunctionHandle * out_array;
    int ret = MXListFunctions(&out_size, &out_array);


}



