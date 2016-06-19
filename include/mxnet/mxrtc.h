/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxrtc.h
 * \brief Wrapper for NVRTC
 * \author Junyuan Xie
 */
#ifndef MXNET_MXRTC_H_
#define MXNET_MXRTC_H_
#include "./base.h"
#if ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
#include <nvrtc.h>
#include <cuda.h>

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "./ndarray.h"

namespace mxnet {

/*!
 * \brief Runtime compile of cuda kernel code with NVRTC
 */
class MXRtc {
 public:
  /*!
   * \brief Build a new kernel.
   *
   * If the same kernel has been compiled before it will be load from
   * cache instead of compile again.
   * \param name name of the kernel function.
   * \param input list of input ndarrays and their name.
   * \param output list of output ndarrays and their name.
   * \param kernel cuda code.
   */
  MXRtc(const std::string& name,
        std::vector<std::pair<std::string, NDArray> > const& input,
        std::vector<std::pair<std::string, NDArray> > const& output,
        const std::string& kernel);
  /*!
   * \brief launch a kernel with the engine.
   * \param input list of input ndarray.
   * \param output list of output ndarray.
   * \param grid_dim_X kernel grid dimensions.
   * \param grid_dim_Y kernel grid dimensions.
   * \param grid_dim_Z kernel grid dimensions.
   * \param block_dim_X kernel block dimensions.
   * \param block_dim_Y kernel block dimensions.
   * \param block_dim_Z kernel block dimensions.
   */
  void push(std::vector<NDArray> const& input,
            std::vector<NDArray> const& output,
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

  /*!
   * \brief add supporting code to kernel.
   */
  std::string decorate(const std::string& name,
                       std::vector<std::pair<std::string, NDArray> > const& input,
                       std::vector<std::pair<std::string, NDArray> > const& output,
                       const std::string kernel);
  /*!
   * \brief compile the kernel with nvrtc.
   */
  char* compile(const std::string& name, const std::string& code);
};

}  // namespace mxnet

#endif  // MXNET_USE_CUDA && MXNET_USE_NVRTC
#endif  // MXNET_MXRTC_H_
