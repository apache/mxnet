
#ifndef mx_generated_function_hpp
#define mx_generated_function_hpp

#include "ndarray.hpp"
#include "mxnet.h"

namespace Rcpp {

class MxFunction1 : public CppFunction {
    public:
        MxFunction1(NDArrayHandle handle, const char * docstring = 0):CppFunction(docstring) {
        
        }

        SEXP operator()(SEXP * args) {
            BEGIN_RCPP

            END_RCPP
        }

    private:
        SEXP (*ptr_fun)();

};

}

#endif
