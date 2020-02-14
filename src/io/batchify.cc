/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file batchify.cc
 * \brief Mini-batch data combination functions.
 */
#include <dmlc/parameter.h>
#include <dmlc/omp.h>
#include <mxnet/io.h>
#include <mshadow/tensor.h>
#include "./inst_vector.h"

namespace mxnet {
namespace io {
struct GroupBatchifyParam : public dmlc::Parameter<GroupBatchifyParam> {
    mxnet::Tuple<std::intptr_t> functions;
    // declare parameters
    DMLC_DECLARE_PARAMETER(GroupBatchifyParam) {
        DMLC_DECLARE_FIELD(functions)
            .describe("Internal sequentially applied batchify functions. "
                      "The number of functions must match output of dataset items.");
    }
};  // struct GroupBatchifyParam
DMLC_REGISTER_PARAMETER(GroupBatchifyParam);

class GroupBatchify : public BatchifyFunction {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      fs_.reserve(param_.functions.ndim());
      for (int i = 0; i < param_.functions.ndim(); ++i) {
          fs_.emplace_back(*static_cast<BatchifyFunctionPtr*>(
              reinterpret_cast<void*>(param_.functions[i])));
      }
    }

    virtual std::vector<TBlob> Batchify(std::vector<std::vector<NDArray> >& inputs) {
      auto out_size = SanityCheck(inputs);
      CHECK_EQ(out_size, fs_.size()) << "In Sequential BatchifyFunction, Elem size "
        << out_size << " and batchify function size " << fs_.size() << " must match";
      std::vector<TBlob> ret;
      ret.reserve(out_size);
      for (size_t i = 0; i < out_size; ++i) {
        std::vector<std::vector<NDArray> > inp;
        inp.reserve(inputs.size());
        for (size_t j = 0; j < inputs.size(); ++j) {
            std::vector<NDArray> curr({inputs[j][i]});
            inp.emplace_back(curr);
        }
        ret.emplace_back(fs_[i]->Batchify(inp)[0]);
      }
      return ret;
    }

  private:
    /*! \brief params */
    GroupBatchifyParam param_;
    /*! \brief internal batchify function pointers */
    std::vector<BatchifyFunctionPtr> fs_;
};  // class GroupBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(GroupBatchify)
  .describe(R"code(Returns the GroupBatchify function.
    )code" ADD_FILELINE)
  .add_arguments(GroupBatchifyParam::__FIELDS__())
  .set_body([]() {
    return new GroupBatchify();
});

struct StackBatchifyParam : public dmlc::Parameter<StackBatchifyParam> {
    /*! \brief Length of the sequence. */
    int use_shared_mem;
    // declare parameters
    DMLC_DECLARE_PARAMETER(StackBatchifyParam) {
        DMLC_DECLARE_FIELD(use_shared_mem).set_default(0)
            .describe("If 1, use shared memory.");
    }
};  // struct StackBatchifyParam

DMLC_REGISTER_PARAMETER(StackBatchifyParam);

class StackBatchify : public BatchifyFunction {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
    }

    virtual std::vector<TBlob> Batchify(std::vector<std::vector<NDArray> >& inputs) {
      auto out_size = SanityCheck(inputs);
      auto bs = inputs.size();
      std::vector<TBlob> ret(out_size);
      for (size_t i = 0; i < out_size; ++i) {
          // Process i-th output
          mxnet::TShape ashape = inputs[0][i].shape();
          CHECK_GE(ashape.ndim(), 0) << "Data dim must be larger than 0";
          // check if all shapes are same
          for (size_t j = 1; j < bs; ++j) {
              CHECK_EQ(ashape, inputs[j][i].shape())
                << "StackBatchify requires all data along batch dim to be the same, "
                << "mismatch " << ashape << " vs. " << inputs[j][i].shape();
          }

          // calculate output ndarray size
          TShape sshape(ashape.ndim() + 1, 0);
          sshape[0] = bs;
          for (int k = 0; k < ashape.ndim(); ++k) {
            sshape[k + 1] = ashape[k];
          }

          // ret[i] = NDArray(sshape, mxnet::Context::CPU(0), false, inputs[0][i].dtype());
          int dtype = inputs[0][i].dtype();
          auto container = new TBlobContainer();
          container->resize(sshape, dtype);
          ret[i] = TBlob(container->dptr_, sshape, cpu::kDevMask, dtype, 0);
          MSHADOW_TYPE_SWITCH_WITH_BOOL(dtype, DType, {
            DType *ptr = ret[i].dptr<DType>();
            auto asize = ashape.Size();
            // _Pragma("omp parallel for num_threads(bs)")
            for (size_t j = 0; j < bs; ++j) {
              inputs[j][i].WaitToRead();
              mshadow::Copy(
                TBlob(ptr + asize * j, inputs[j][i].data().shape_, cpu::kDevMask, dtype, 0).FlatTo2D<cpu, DType>(),
                inputs[j][i].data().FlatTo2D<cpu, DType>());
              // std::memcpy(ptr + asize * j, inputs[j][i].data().dptr<DType>(), asize * sizeof(DType));
            }
          })
      }
      return ret;
    }
  private:
    /*! \brief parameters */
    StackBatchifyParam param_;
    /*! \brief OMPException obj to store and rethrow exceptions from omp blocks*/
    dmlc::OMPException omp_exc_;
};  // class StackBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(StackBatchify)
  .describe(R"code(Returns the StackBatchify function.
    )code" ADD_FILELINE)
  .add_arguments(StackBatchifyParam::__FIELDS__())
  .set_body([]() {
    return new StackBatchify();
});
}  // namespace io
}  // namespace mxnet
