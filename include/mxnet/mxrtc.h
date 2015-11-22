/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxrtc.h
 * \brief Wrapper for NVRTC
 * \author Junyuan Xie
 */
#ifndef MXNET_MXRTC_H_
#define MXNET_MXRTC_H_
#include "./base.h"
#if MXNET_USE_CUDA

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
              unsigned int  grid_dim_X,
              unsigned int  grid_dim_Y,
              unsigned int  grid_dim_Z,
              unsigned int  block_dim_X,
              unsigned int  block_dim_Y,
              unsigned int  block_dim_Z);

 private:
    static const std::string str_type;
    static std::unordered_map<std::string, char*> kernel_registry;

    std::string name_;
    index_t num_input_, num_output_;
    std::string code_;
    char* ptx_;
    std::unordered_map<int, CUmodule> module_;
    std::unordered_map<int, CUfunction> func_;

    std::string decorate(const std::string& name,
                         std::vector<std::pair<std::string, NDArray*> > const& input,
                         std::vector<std::pair<std::string, NDArray*> > const& output,
                         const std::string kernel);
    char* compile(const std::string& name, const std::string& code);
};

}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_MXRTC_H_
