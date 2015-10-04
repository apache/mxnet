
#ifndef Rcpp_ndarray_hpp
#define Rcpp_ndarray_hpp

#include <Rcpp.h>
#include "mxnet.h"

class NDArray {
    public:
        NDArray(NDArrayHandle handle, bool writable = true): handle(handle), writable(writable){}
        NDArray(const NDArray& n):handle(n.handle), writable(n.writable){}
        void load(const std::string & filename);
        void save(const std::string & filename);

        NDArrayHandle handle;
        bool writable;
};

#endif
