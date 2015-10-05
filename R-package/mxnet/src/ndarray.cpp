
#include "ndarray.hpp"
#include "mx_generated_function.hpp"

void save(SEXP sxptr, const std::string & filename) {
    if (TYPEOF(sxptr) == 19) {
        Rcpp::List data_lst(sxptr);
        std::vector<std::string> lst_names = data_lst.names();
        int num_args = data_lst.size();
        NDArrayHandle * handles = (NDArrayHandle *)malloc(num_args * sizeof(NDArrayHandle));
        std::vector<const char*> keys;
        for (int i = 0 ; i < num_args; i++) {
            keys.push_back(lst_names[i].c_str());
            Rcpp::XPtr<NDArray> * ptr = new Rcpp::XPtr<NDArray>(sxptr);
            handles[i] = (*(ptr->get())).handle;
        }
        MXNDArraySave(filename.c_str(), num_args, handles, &keys[0]);
    } else if (TYPEOF(sxptr) == 22) {
        Rcpp::XPtr<NDArray> ptr(sxptr);
        NDArray data = *ptr.get();
        MXNDArraySave(filename.c_str(), 1, &data.handle, NULL);
    } else {
        Rcpp::Rcerr << "only NDArray or list of NDArray" << std::endl;
    }
}

SEXP load(const std::string & filename) {
    mx_uint out_size;
    NDArrayHandle* out_arr;
    mx_uint out_name_size;
    const char** out_names;
    MXNDArrayLoad(filename.c_str(), &out_size, &out_arr, &out_name_size, &out_names);
    std::vector<std::string> lst_names(out_size);
    Rcpp::List out(out_size);
    for (int i = 0; i < out_size; i++) {
        out[i] = Rcpp::XPtr<NDArray>(new NDArray(out_arr[i]));
    }

    for (int i = 0; i < out_size; i++) {
        if (out_name_size != 0)
            lst_names[i] = out_names[i];
        else
            lst_names[i] = "X" + std::to_string(i);
    }
    out.attr("names") = lst_names;
    return out;
}

RCPP_MODULE(mod_ndarray) {
    using namespace Rcpp;
    function("load_ndarray", &load);
    function("save_ndarray", &save);

    mx_uint out_size;
    FunctionHandle * out_array;
    int ret = MXListFunctions(&out_size, &out_array);

    for (mx_uint i = 0; i < out_size; i++) {
        MxFunction * fun = new MxFunction(out_array[i]);
        _rcpp_module_mod_ndarray.Add(fun->get_name(), fun);
    }

}

