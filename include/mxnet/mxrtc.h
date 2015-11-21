/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxrtc.h
 * \brief Wrapper for NVRTC
 * \author Junyuan Xie
 */
#ifndef MXNET_MXRTC_H_
#define MXNET_MXRTC_H_

#include <nvrtc.h>
#include <cuda.h>

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "./ndarray.h"



namespace mxnet {


class MXRtc {
 public:
    MXRtc(const std::string& name,
          std::vector<std::pair<std::string, NDArray*> > const& input,
          std::vector<std::pair<std::string, NDArray*> > const& output,
          const std::string& kernel);
    void push(std::vector<NDArray*> const& input,
              std::vector<NDArray*> const& output,
              unsigned int  gridDimX,
              unsigned int  gridDimY,
              unsigned int  gridDimZ,
              unsigned int  blockDimX,
              unsigned int  blockDimY,
              unsigned int  blockDimZ);

 private:
    static const std::string str_type;
    static std::unordered_map<std::string, char*> kernel_registry;

    std::string name_;
    index_t num_input_, num_output_;
    std::string code_;
    char* ptx_;
    CUmodule module_;
    CUfunction func_;

    std::string decorate(const std::string& name,
                         std::vector<std::pair<std::string, NDArray*> > const& input,
                         std::vector<std::pair<std::string, NDArray*> > const& output,
                         const std::string kernel);
    char* compile(const std::string& name, const std::string& code);
};

}  // namespace mxnet
#endif  // MXNET_MXRTC_H_
