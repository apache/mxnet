
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

NDArray make_ndarray_function(NDArrayHandle handle);

NDArray binary_ndarray_function(NDArray lhs, NDArray rhs, NDArray out);
