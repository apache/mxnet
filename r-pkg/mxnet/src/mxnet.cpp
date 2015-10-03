
#include <Rcpp.h>

#include "c_api.h"

using namespace Rcpp;

//[[Rcpp::export]]
List rcpp_ndarray_load(std::string filename) {

    mx_uint out_size;
    mx_uint out_name_size;
    NDArrayHandle * handles;
    const char** names;
    if (MXNDArrayLoad(filename.c_str(), 
                      &out_size,
                      &handles,
                      &out_name_size,
                      &names) != 0)
        return R_NilValue;   
    
}
