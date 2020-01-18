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
#include "./batchify.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::BatchifyFunctionReg);
}  // namespace dmlc

namespace mxnet {
namespace io {
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
    std::vector<NDArray> Batchify(std::vector<std::vector<NDArray> > inputs,
                                  std::vector<int> keep_dim) {
      auto bs = inputs.size();
      CHECK_GT(bs, 0) << "BatchifyFunction should handle at lease 1 sample";
      auto out_size = inputs[0].size();
      // sanity check: each input has same size
      for (size_t i = 1; i < bs; ++i) {
          CHECK_EQ(inputs[i].size(), out_size)
            << i << "-th input size does not match " << out_size;
      }
      CHECK_EQ(out_size, keep_dim.size())
        << "inputs and keep_dim size mismatch "
        << out_size << " vs. " << keep_dim.size();
      std::vector<NDArray> ret(out_size);
      
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
          for (int k = 0; k < ashape.ndim(); ++i) {
            sshape[k + 1] = ashape[k];
          }
          
          ret[i].ReshapeAndAlloc(sshape);
          for (size_t j = 0; j < bs; ++j) {
              auto slice_view = ret[i].Slice(j, j + 1);
              slice_view.SyncCopyFromNDArray(inputs[j][i]);
          }

          // reshape if keep_dim is true
          if (keep_dim[i] == 1 && sshape.ndim() > 1) {
            TShape new_shape = ashape;
            new_shape[0] *= bs;
            ret[i].Reshape(new_shape);
          }
      }
      return ret;
    }
};  // class StackBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(StackBatchify);
}  // namespace io
}  // namespace mxnet