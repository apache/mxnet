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
 *  Copyright (c) 2020 by Contributors
 * \file batchify.cc
 * \brief Mini-batch data combination functions.
 */
#include <dmlc/parameter.h>
#include <dmlc/omp.h>
#include <mxnet/io.h>
#include <mshadow/tensor.h>
#include <mshadow/extension.h>
#include <mshadow/extension/slice.h>

#include <stack>
#include <cmath>

#include "./inst_vector.h"
#include "../ndarray/ndarray_function.h"

namespace mxnet {
namespace io {


#define tostr(s) #s

#ifdef _MSC_VER
  #if _MSC_VER < 1925
    #define omp_parallel(t) __pragma(omp parallel for num_threads(t))
  #else
    #define omp_parallel(t) _Pragma(tostr(omp parallel for num_threads( ## t ## )))
  #endif
#else
  #define omp_parallel(t) _Pragma(tostr(omp parallel for num_threads(t)))
#endif

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
  explicit GroupBatchify(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    fs_.reserve(param_.functions.ndim());
    for (int i = 0; i < param_.functions.ndim(); ++i) {
        fs_.emplace_back(*static_cast<BatchifyFunctionPtr*>(
            reinterpret_cast<void*>(param_.functions[i])));
    }
  }

  bool Batchify(const std::vector<std::vector<NDArray> >& inputs,
                        std::vector<NDArray>* outputs) override {
    auto bs = inputs.size();
    CHECK_GT(bs, 0) << "BatchifyFunction should handle at lease 1 sample";
    auto out_size = inputs[0].size();
    CHECK_EQ(out_size, fs_.size()) << "In GroupBatchifyFunction, Elem size "
      << out_size << " and batchify function size " << fs_.size() << " must match";
    outputs->resize(out_size);
    for (size_t i = 0; i < out_size; ++i) {
      std::vector<std::vector<NDArray> > inp;
      inp.reserve(inputs.size());
      for (const auto & input : inputs) {
          std::vector<NDArray> curr({input[i]});
          inp.emplace_back(curr);
      }
      std::vector<NDArray> tmp;
      if (!fs_[i]->Batchify(inp, &tmp)) return false;
      (*outputs)[i] = tmp[0];
    }
    return true;
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
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
    return new GroupBatchify(kwargs);
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
  explicit StackBatchify(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
  }

  bool Batchify(const std::vector<std::vector<NDArray> >& inputs,
                        std::vector<NDArray>* outputs) override {
    auto out_size = SanityCheck(inputs);
    auto bs = inputs.size();
    outputs->resize(out_size);
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

        int dtype = inputs[0][i].dtype();
        if (!(*outputs)[i].is_none() && (*outputs)[i].ctx() == mxnet::Context::CPU(0) &&
            (*outputs)[i].dtype() == dtype &&
            (*outputs)[i].storage_type() == kDefaultStorage) {
          if ((*outputs)[i].shape() != sshape) {
            // realloc
            (*outputs)[i].ReshapeAndAlloc(sshape);
          }
        } else {
          (*outputs)[i] = NDArray(sshape, mxnet::Context::CPU(0), false, inputs[0][i].dtype());
        }
        int sbs = static_cast<int>(bs);
        MSHADOW_TYPE_SWITCH_WITH_BOOL(dtype, DType, {
          omp_parallel(bs)
          for (int j = 0; j < sbs; ++j) {
            omp_exc_.Run([&] {
              // inputs[j][i].WaitToRead();
              DType *ptr = (*outputs)[i].data().dptr<DType>();
              auto asize = ashape.Size();
              RunContext rctx{(*outputs)[i].ctx(), nullptr, nullptr, false};
              auto dst = TBlob(
                ptr + asize * j, inputs[j][i].data().shape_, cpu::kDevMask, dtype, 0);
              mxnet::ndarray::Copy<cpu, cpu>(
                inputs[j][i].data(), &dst, Context::CPU(), Context::CPU(), rctx);
            });
          }
          omp_exc_.Rethrow();
        })
    }
    return true;
  }
 private:
  /*! \brief parameters */
  StackBatchifyParam param_;
  /*! \brief OMPException obj to store and rethrow exceptions from omp blocks*/
  dmlc::OMPException omp_exc_;

  std::size_t SanityCheck(const std::vector<std::vector<NDArray> >& inputs) {
    auto bs = inputs.size();
    CHECK_GT(bs, 0) << "BatchifyFunction should handle at lease 1 sample";
    auto out_size = inputs[0].size();
    // sanity check: each input has same size
    for (size_t i = 1; i < bs; ++i) {
        CHECK_EQ(inputs[i].size(), out_size)
          << i << "-th input size does not match " << out_size;
    }
    return out_size;
  }
};  // class StackBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(StackBatchify)
  .describe(R"code(Returns the StackBatchify function.
    )code" ADD_FILELINE)
  .add_arguments(StackBatchifyParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
    return new StackBatchify(kwargs);
});

struct PadBatchifyParam : public dmlc::Parameter<PadBatchifyParam> {
  int use_shared_mem;
  double pad_val;
  int dtype;
  int round_to;
  // declare parameters
  DMLC_DECLARE_PARAMETER(PadBatchifyParam) {
      DMLC_DECLARE_FIELD(use_shared_mem).set_default(0)
          .describe("If 1, use shared memory.");
      DMLC_DECLARE_FIELD(pad_val).set_default(0)
          .describe("The filled values, default to 0.");
      DMLC_DECLARE_FIELD(dtype).set_default(-1)
        .describe("If not -1, force to use dtype as output type, otherwise use input type.");
      DMLC_DECLARE_FIELD(round_to).set_default(-1)
        .describe("If > 0, the padded dimension will be rounded to be multiple of this value.");
  }
};  // struct PadBatchifyParam

DMLC_REGISTER_PARAMETER(PadBatchifyParam);

class PadBatchify : public BatchifyFunction {
 public:
  explicit PadBatchify(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
  }

  bool Batchify(const std::vector<std::vector<NDArray> >& inputs,
                        std::vector<NDArray>* outputs) override {
    auto bs = inputs.size();
    CHECK_GT(bs, 0) << "BatchifyFunction should handle at lease 1 sample";
    auto out_size = inputs[0].size();
    outputs->resize(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        // Process i-th output
        mxnet::TShape ashape = inputs[0][i].shape();
        CHECK_GE(ashape.ndim(), 0) << "Data dim must be larger than 0";
        // find the maximum size in each dim
        for (size_t j = 1; j < bs; ++j) {
          mxnet::TShape other_shape = inputs[j][i].shape();
          CHECK_EQ(ashape.ndim(), other_shape.ndim())
            << "PadBatchify expects all inputs to have same dimensionality: given "
            << ashape.ndim() << " vs. " << other_shape.ndim();
            for (dim_t k = 0; k < ashape.ndim(); ++k) {
              ashape[k] = std::max(ashape[k], other_shape[k]);
            }
        }
        for (dim_t k = 0; k < ashape.ndim(); ++k) {
          // pad to multiple of round_to
          if (param_.round_to > 0) {
            ashape[k] = param_.round_to * static_cast<int>(
              std::ceil(static_cast<double>(ashape[k] / param_.round_to)));
          }
        }

        // calculate output ndarray size
        TShape sshape(ashape.ndim() + 1, 0);
        sshape[0] = bs;
        for (int k = 0; k < ashape.ndim(); ++k) {
          sshape[k + 1] = ashape[k];
        }

        int dtype = param_.dtype > -1 ? param_.dtype : inputs[0][i].dtype();
        if (!(*outputs)[i].is_none() &&
            (*outputs)[i].ctx() == mxnet::Context::CPU(0) &&
            (*outputs)[i].dtype() == dtype &&
            (*outputs)[i].storage_type() == kDefaultStorage) {
          if ((*outputs)[i].shape() != sshape) {
            // realloc
            (*outputs)[i].ReshapeAndAlloc(sshape);
          }
        } else {
          (*outputs)[i] = NDArray(sshape, mxnet::Context::CPU(0), false, inputs[0][i].dtype());
        }
        MSHADOW_TYPE_SWITCH_WITH_BOOL(dtype, DType, {
          // fill pad value first
          std::fill((*outputs)[i].data().dptr<DType>(),
                    (*outputs)[i].data().dptr<DType>() + sshape.Size(),
                    static_cast<DType>(param_.pad_val));
          DType *ptr = (*outputs)[i].data().dptr<DType>();
          auto asize = ashape.Size();
          int sbs = static_cast<int>(bs);
          omp_parallel(bs)
          for (int j = 0; j < sbs; ++j) {
            using namespace mshadow::expr;
            auto compact_shapes = CompactShapes(ashape, inputs[j][i].shape());
            // inputs[j][i].WaitToRead();
            auto& fshape = compact_shapes.first;
            auto& cshape = compact_shapes.second;
            switch (fshape.size()) {
              case 1U: {
                mshadow::Tensor<cpu, 1, DType> dst = TBlob(
                  ptr + asize * j, ashape, cpu::kDevMask, dtype, 0).get_with_shape<cpu, 1, DType>(
                    mshadow::Shape1(fshape[0]));
                mshadow::Tensor<cpu, 1, DType> src = inputs[j][i].data().get_with_shape<
                  cpu, 1, DType>(mshadow::Shape1(cshape[0]));
                slice<0>(dst, 0, cshape[0]) = src;
                break;
              }
              case 2U: {
                mshadow::Tensor<cpu, 2, DType> dst = TBlob(
                  ptr + asize * j, ashape, cpu::kDevMask, dtype, 0).get_with_shape<cpu, 2, DType>(
                    mshadow::Shape2(fshape[0], fshape[1]));
                mshadow::Tensor<cpu, 2, DType> src = inputs[j][i].data().get_with_shape<
                  cpu, 2, DType>(mshadow::Shape2(cshape[0], cshape[1]));
                slice<1>(slice<0>(dst, 0, cshape[0]), 0, cshape[1]) = src;
                break;
              }
              case 3U: {
                mshadow::Tensor<cpu, 3, DType> dst = TBlob(
                  ptr + asize * j, ashape, cpu::kDevMask, dtype, 0).get_with_shape<cpu, 3, DType>(
                    mshadow::Shape3(fshape[0], fshape[1], fshape[2]));
                mshadow::Tensor<cpu, 3, DType> src = inputs[j][i].data().get_with_shape<
                  cpu, 3, DType>(mshadow::Shape3(cshape[0], cshape[1], cshape[2]));
                slice<2>(slice<1>(slice<0>(dst, 0, cshape[0]), 0, cshape[1]), 0, cshape[2]) = src;
                break;
              }
              case 4U: {
                mshadow::Tensor<cpu, 4, DType> dst = TBlob(
                  ptr + asize * j, ashape, cpu::kDevMask, dtype, 0).get_with_shape<cpu, 4, DType>(
                    mshadow::Shape4(fshape[0], fshape[1], fshape[2], fshape[3]));
                mshadow::Tensor<cpu, 4, DType> src = inputs[j][i].data().get_with_shape<
                  cpu, 4, DType>(mshadow::Shape4(cshape[0], cshape[1], cshape[2], cshape[3]));
                slice<3>(slice<2>(slice<1>(slice<0>(dst, 0, cshape[0]), 0, cshape[1]),
                                                    0, cshape[2]), 0, cshape[3]) = src;
                break;
              }
              case 5U: {
                mshadow::Tensor<cpu, 5, DType> dst = TBlob(
                  ptr + asize * j, ashape, cpu::kDevMask, dtype, 0).get_with_shape<cpu, 5, DType>(
                    mshadow::Shape5(fshape[0], fshape[1], fshape[2], fshape[3], fshape[4]));
                mshadow::Tensor<cpu, 5, DType> src = inputs[j][i].data().get_with_shape<
                  cpu, 5, DType>(mshadow::Shape5(
                    cshape[0], cshape[1], cshape[2], cshape[3], cshape[4]));
                slice<4>(slice<3>(slice<2>(slice<1>(slice<0>(
                  dst, 0, cshape[0]), 0, cshape[1]), 0, cshape[2]),
                  0, cshape[3]), 0, cshape[4]) = src;
                break;
              }
              default: {
                LOG(FATAL) << "# dim to pad: " << cshape.size() << " exceeds limit of 5.";
              }
            }
          }
        })
    }
    return true;
  }

 private:
  /*! \brief parameters */
  PadBatchifyParam param_;
  /*! \brief OMPException obj to store and rethrow exceptions from omp blocks*/
  dmlc::OMPException omp_exc_;

  std::pair<std::vector<dim_t>, std::vector<dim_t>> CompactShapes(const TShape& ashape,
                                                                  const TShape& ishape) {
    // squeeze dimensions that do not need pad
    std::stack<dim_t> dim_stack;
    std::vector<dim_t> full_shape;
    std::vector<dim_t> data_shape;
    for (dim_t k = 0; k < ishape.ndim(); ++k) {
      if (ishape[k] == ashape[k]) {
        dim_stack.push(ishape[k]);
      } else {
        dim_t ss = 1;
        while (!dim_stack.empty()) {
          ss *= dim_stack.top();
          dim_stack.pop();
        }
        if (ss > 1) {
          full_shape.emplace_back(ss);
          data_shape.emplace_back(ss);
        }
        full_shape.emplace_back(ashape[k]);
        data_shape.emplace_back(ishape[k]);
      }
    }
    // clear the stack
    index_t ss = 1;
    while (!dim_stack.empty()) {
      ss *= dim_stack.top();
      dim_stack.pop();
    }
    if (ss > 1 || full_shape.empty()) {
      full_shape.emplace_back(ss);
      data_shape.emplace_back(ss);
    }
    CHECK_EQ(full_shape.size(), data_shape.size());
    CHECK_GE(data_shape.size(), 1U);
    return std::make_pair(full_shape, data_shape);
  }
};  // class PadBatchify

MXNET_REGISTER_IO_BATCHIFY_FUNCTION(PadBatchify)
  .describe(R"code(Returns the StackBatchify function.
    )code" ADD_FILELINE)
  .add_arguments(PadBatchifyParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
    return new PadBatchify(kwargs);
});
}  // namespace io
}  // namespace mxnet
