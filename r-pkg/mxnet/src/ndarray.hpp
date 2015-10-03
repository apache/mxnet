
#ifndef Rcpp_ndarray_hpp
#define Rcpp_ndarray_hpp

#include <Rcpp.h>
#include "mxnet.h"

class NDArray {
    public:
        void load(const std::string & filename);
        void save(const std::string & filename);
    private:
        NDArrayHandle handle;
        bool writable;
};

#endif
