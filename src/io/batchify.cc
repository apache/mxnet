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
#include <mxnet/io.h>

namespace mxnet {
namespace io {
struct SequentialBatchifyParam : public dmlc::Parameter<SequentialBatchifyParam> {
    mxnet::Tuple<std::intptr_t> functions;
    // declare parameters
    DMLC_DECLARE_PARAMETER(SequentialBatchifyParam) {
        DMLC_DECLARE_FIELD(functions)
            .describe("Internal sequentially applied batchify functions. "
                      "The number of functions must match output of dataset items.");
    }
};  // struct SequentialBatchifyParam
DMLC_REGISTER_PARAMETER(SequentialBatchifyParam);

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

class SequentialBatchify : public BatchifyFunction {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      fs_.reserve(param_.functions.ndim());
      for (int i = 0; i < param_.functions.ndim(); ++i) {
          fs_.emplace_back(static_cast<BatchifyFunction*>(
              reinterpret_cast<void*>(param_.functions[i])));
      }
    }

    virtual std::vector<NDArray> Batchify(std::vector<std::vector<NDArray> >& inputs) {
      auto out_size = SanityCheck(inputs);
      CHECK_EQ(out_size, fs_.size()) << "In Sequential BatchifyFunction, Elem size "
        << out_size << " and batchify function size " << fs_.size() << " must match";
      std::vector<NDArray> ret;
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
    SequentialBatchifyParam param_;
    /*! \brief internal batchify function pointers */
    std::vector<BatchifyFunction*> fs_;
};  // class SequentialBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(SequentialBatchify)
  .describe(R"code(Returns the SequentialBatchify function.
    )code" ADD_FILELINE)
  .add_arguments(SequentialBatchifyParam::__FIELDS__())
  .set_body([]() {
    return new SequentialBatchify();
});

class StackBatchify : public BatchifyFunction {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
    }

    virtual std::vector<NDArray> Batchify(std::vector<std::vector<NDArray> >& inputs) {
      LOG(INFO) << "Entered batchify";
      auto out_size = SanityCheck(inputs);
      auto bs = inputs.size();
      std::vector<NDArray> ret(out_size);
      LOG(INFO) << "out size: " << out_size;

      #pragma omp parallel num_threads(out_size)
      for (size_t i = 0; i < out_size; ++i) {
          // Process i-th output
          auto ashape = inputs[0][i].shape();
          CHECK_GE(ashape.ndim(), 1) << "Data dim must be larger than 1";
          // check if all shapes are same
          for (size_t j = 1; j < bs; ++j) {
              CHECK_EQ(ashape, inputs[j][i].shape())
                << "StackBatchify requires all data along batch dim to be the same, "
                << "mismatch " << ashape << " vs. " << inputs[j][i].shape();
          }

          // calculate output ndarray size
          TShape sshape = TShape(ashape.ndim() + 1, 0);
          sshape[0] = bs;
          for (int k = 0; k < ashape.ndim(); ++k) {
            sshape[k + 1] = ashape[k];
          }

          ret[i] = NDArray(sshape, mxnet::Context::CPU(0), false, inputs[0][i].dtype());
          for (size_t j = 0; j < bs; ++j) {
              auto slice_view = ret[i].Slice(j, j + 1);
              slice_view.SyncCopyFromNDArray(inputs[j][i]);
          }

          // reshape to keep dim
          if (sshape.ndim() > 1) {
            TShape new_shape = ashape;
            new_shape[0] *= bs;
            ret[i].Reshape(new_shape);
          }
      }
      return ret;
    }
  private:
    /*! \brief parameters */
    StackBatchifyParam param_;
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
