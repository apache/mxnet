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
 *  Copyright (c) 2019 by Contributors
 * \file np_insert_op-inl.h
 * \brief Function definition of insert operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_

#include <vector>
#include <memory>
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct NumpyInsertParam : public dmlc::Parameter<NumpyInsertParam> {
  dmlc::optional<int> start;
  dmlc::optional<int> stop;
  dmlc::optional<int> step;
  dmlc::optional<int> int_ind;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyInsertParam) {
    DMLC_DECLARE_FIELD(start)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'start' is one of it's arguments.");
    DMLC_DECLARE_FIELD(stop)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'stop' is one of it's arguments.");
    DMLC_DECLARE_FIELD(step)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'step' is one of it's arguments.");
    DMLC_DECLARE_FIELD(int_ind)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is int, 'int_ind' is the index before which"
              "'values' is inserted");
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("Axis along which to insert `values`.");
  }
};

namespace insert_ {
enum InsertOpInputs {kArr, kValues, kObj};
enum InsertOpOutputs {kOut};
}  // namespace insert_

template<int req>
struct InsertZeroNdimForward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<int req>
struct InsertSingleIndexForward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const DType* in_val, const DType* in_arr,
                                  const mshadow::Shape<10> outshape,
                                  const mshadow::Shape<10> valshape,
                                  const int index, const int numnew,
                                  const mshadow::Shape<10> val_stride,
                                  const mshadow::Shape<10> old_val_stride,
                                  const mshadow::Shape<10> arr_stride,
                                  const mshadow::Shape<10> out_stride,
                                  const int arr_ndim, const int val_ndim,
                                  const int out_ndim, const int axis,
                                  bool moveaxis) {
    const int64_t out_head = i / out_stride[axis];
    const int64_t out_mid = out_head % outshape[axis];
    mshadow::Shape<10> out_idx;  // i -> position in output's shape
    for (int j = 0; j < out_ndim; ++j) {
      const int64_t head = i / out_stride[j];
      const int64_t mid = head % outshape[j];
      out_idx[j] = mid;
    }
    int64_t dest_idx;
    if (out_mid >= index && out_mid < index + numnew) {
      int idx_val = out_mid - index;
      mshadow::Shape<10> val_idx(out_idx);  // i -> position in values's shape
      val_idx[axis] = idx_val;
      for (int j = out_ndim - 1, k = val_ndim - 1; j >= 0 || k >= 0; --j, --k) {
        if (j >= 0 && k >= 0) {
          if (valshape[k] == 1) {
            val_idx[k] = 0;
          }
        } else if (j >= 0) {
          val_idx[j] = 1;
        } else {
          break;
        }
      }
      dest_idx = 0;
      if (moveaxis) {
        for (int _i = 0; _i < axis; ++_i) {
          dest_idx += old_val_stride[_i + 1] * val_idx[_i];
        }
        dest_idx += old_val_stride[0] * val_idx[axis];
        for (int _i = axis + 1; _i < val_ndim ; ++_i) {
          dest_idx += old_val_stride[_i] *val_idx[_i];
        }
      } else {
        for (int _i =0; _i < val_ndim; ++_i) {
          dest_idx += val_stride[_i] * val_idx[_i];
        }
      }
      KERNEL_ASSIGN(out_data[i], req, in_val[dest_idx]);
    } else {
      int idx_arr = (out_mid < index) ? out_mid : out_mid - numnew;
      mshadow::Shape<10> arr_idx(out_idx);  // i -> position in arr's shape
      arr_idx[axis] = idx_arr;
      dest_idx = 0;
      for (int _i =0; _i < arr_ndim; ++_i) {
        dest_idx += arr_stride[_i] * arr_idx[_i];
      }
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }

  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const DType* in_val, const DType* in_arr,
                                  const mshadow::Shape<10> outshape,
                                  const mshadow::Shape<10> valshape,
                                  const int N, const IType* in_obj, const int numnew,
                                  const mshadow::Shape<10> val_stride,
                                  const mshadow::Shape<10> old_val_stride,
                                  const mshadow::Shape<10> arr_stride,
                                  const mshadow::Shape<10> out_stride,
                                  const int arr_ndim, const int val_ndim,
                                  const int out_ndim, const int axis,
                                  bool moveaxis) {
    const int64_t out_head = i / out_stride[axis];
    const int64_t out_mid = out_head % outshape[axis];
    mshadow::Shape<10> out_idx;  // i -> position in output's shape
    for (int j = 0; j < out_ndim; ++j) {
      const int64_t head = i / out_stride[j];
      const int64_t mid = head % outshape[j];
      out_idx[j] = mid;
    }
    int64_t dest_idx;
    IType index = in_obj[0];
    if (static_cast<int64_t>(index) < 0) {
        index += static_cast<IType>(N);
    }
    if (out_mid >= index && out_mid < index + numnew) {
      int idx_val = out_mid - index;
      mshadow::Shape<10> val_idx(out_idx);
      val_idx[axis] = idx_val;
      for (int j = out_ndim - 1, k = val_ndim - 1; j >= 0 || k >= 0; --j, --k) {
        if (j >= 0 && k >= 0) {
          if (valshape[k] == 1) {
            val_idx[k] = 0;
          }
        } else if (j >= 0) {
          val_idx[j] = 1;
        } else {
          break;
        }
      }
      dest_idx = 0;
      if (moveaxis) {
        for (int _i = 0; _i < axis; ++_i) {
          dest_idx += old_val_stride[_i + 1] * val_idx[_i];
        }
        dest_idx += old_val_stride[0] * val_idx[axis];
        for (int _i = axis + 1; _i < val_ndim ; ++_i) {
          dest_idx += old_val_stride[_i] *val_idx[_i];
        }
      } else {
        for (int _i =0; _i < val_ndim; ++_i) {
          dest_idx += val_stride[_i] * val_idx[_i];
        }
      }
      KERNEL_ASSIGN(out_data[i], req, in_val[dest_idx]);
    } else {
      int idx_arr = (out_mid < index) ? out_mid : out_mid - numnew;
      mshadow::Shape<10> arr_idx(out_idx);  // i -> position in arr's shape
      arr_idx[axis] = idx_arr;
      dest_idx = 0;
      for (int _i =0; _i < arr_ndim; ++_i) {
        dest_idx += arr_stride[_i] * arr_idx[_i];
      }
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }
};

template<int req>
struct InsertSeqForward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const DType* in_val, const DType* in_arr,
                                  const mshadow::Shape<10> outshape,
                                  const mshadow::Shape<10> valshape,
                                  const int* is_insert,
                                  const int* origin_idx,
                                  const mshadow::Shape<10> val_stride,
                                  const mshadow::Shape<10> arr_stride,
                                  const mshadow::Shape<10> out_stride,
                                  const int arr_ndim, const int val_ndim,
                                  const int out_ndim, const int axis) {
    const int64_t out_head = i / out_stride[axis];
    const int64_t out_mid = out_head % outshape[axis];
    mshadow::Shape<10> out_idx;  // i -> position in output's shape
    for (int j = 0; j < out_ndim; ++j) {
      const int64_t head = i / out_stride[j];
      const int64_t mid = head % outshape[j];
      out_idx[j] = mid;
    }
    int64_t dest_idx;
    if (is_insert[out_mid]) {
      int idx_val = origin_idx[out_mid];
      mshadow::Shape<10> insert_idx(out_idx);  // i -> position in insert's shape
      insert_idx[axis] = idx_val;
      mshadow::Shape<10> val_idx(insert_idx);  // i -> position in values's shape
      for (int j = out_ndim - 1, k = val_ndim - 1; j >= 0 || k >= 0; --j, --k) {
        if (j >= 0 && k >= 0) {
          if (valshape[k] == 1) {
            val_idx[k] = 0;
          }
        } else if (j >= 0) {
          val_idx[j] = 0;
        } else {
          break;
        }
      }
      dest_idx = 0;
      for (int _i =0; _i < val_ndim; ++_i) {
        dest_idx += val_stride[_i] * val_idx[_i];
      }
      KERNEL_ASSIGN(out_data[i], req, in_val[dest_idx]);
    } else {
      int idx_arr = origin_idx[out_mid];
      mshadow::Shape<10> arr_idx(out_idx);  // i -> position in arr's shape
      arr_idx[axis] = idx_arr;
      dest_idx = 0;
      for (int _i =0; _i < arr_ndim; ++_i) {
        dest_idx += arr_stride[_i] * arr_idx[_i];
      }
      out_data[i] = in_arr[dest_idx];
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }
};

struct SliceToIndices {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, int N,
                                  int start, int step) {
    indices[i] = start + i * step;
    if (static_cast<int64_t>(indices[i]) < 0) {
      indices[i] += static_cast<IType>(N);
    }
  }
};

struct ObjToIndices {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices,
                                  int N, const IType* obj) {
    indices[i] = obj[i];
    if (static_cast<int64_t>(indices[i]) < 0) {
      indices[i] += static_cast<IType>(N);
    }
  }
};

struct AssignId {
  MSHADOW_XINLINE static void Map(int i, int* order) {
    order[i] = i;
  }
};

struct IndicesModify {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, const int* order) {
    indices[order[i]] += i;
  }
};

struct AssignInsertZero {
  MSHADOW_XINLINE static void Map(int i, int* is_insert) {
    is_insert[i] = 0;
  }
};

struct SetIsInsert {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, int* is_insert) {
    is_insert[static_cast<int>(indices[i])] = 1;
  }
};

struct SetOriginValuesIdx {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, const IType* indices, int* origin_idx) {
    origin_idx[static_cast<int>(indices[i])] = i;
  }
};

struct SetOriginArrIdx {
  MSHADOW_XINLINE static void Map(int i, const int* is_insert,
                         int* origin_idx) {
    if (!is_insert[i]) {
      int cnt = 0;
      for (int j = 0; j < i; ++j) {
        if (is_insert[j] == 0) {
          cnt++;
        }
      }
      origin_idx[i] = cnt;
    }
  }
};

template<typename xpu>
void NumpyInsertCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mxnet_op;

    const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
    CHECK_EQ(inputs.size(),
            (param.stop.has_value() || param.int_ind.has_value()) ? 2U : 3U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    int ndim = inputs[insert_::kArr].shape_.ndim();
    int axis = param.axis.has_value() ? param.axis.value() : 0;
    TBlob arr, values;
    if (!param.axis.has_value()) {
        arr = inputs[insert_::kArr].reshape(Shape1(inputs[insert_::kArr].shape_.Size()));
        ndim = 1;
    } else if (ndim == 0) {
        arr = inputs[insert_::kArr];
        CHECK_EQ(inputs[insert_::kValues].shape_.ndim(), 0)
            << "'arr' is a 0-d array, 'values' can not assign to it. "
            << "alueError: assignment to 0-d array.";
        MSHADOW_TYPE_SWITCH(outputs[insert_::kOut].type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[insert_::kOut], req_type, {
                Kernel<InsertZeroNdimForward<req_type>, xpu>::Launch(
                    s, outputs[insert_::kOut].shape_.Size(),
                    outputs[insert_::kOut].dptr<DType>(), inputs[insert_::kValues].dptr<DType>());
            });
        });
        return;
    } else {
        arr = inputs[insert_::kArr];
        CHECK(axis >= -1 * arr.shape_.ndim() && axis < arr.shape_.ndim())
          << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
        axis += (axis < 0) ? arr.shape_.ndim() : 0;
    }

    int N = arr.shape_[axis];
    mxnet::TShape newshape(arr.shape_);
    size_t indices_len = 0;
    int start = 0, stop = 0, step = 0;

    // get and check indices from slice or sequence of ints
    if (inputs.size() == 3U) {
        indices_len = inputs[insert_::kObj].shape_.Size();
    } else if (param.stop.has_value()) {
        step = param.step.value();
        CHECK_NE(step, 0) << "'step' can not equal to 0.";
        stop = param.stop.value();
        stop += (stop < 0) ? N : 0;
        stop = (stop < 0) ? ((step < 0) ? -1 : 0) : stop;
        stop = (stop >= N) ? ((step < 0) ? N - 1 : N) : stop;
        start = param.start.value();
        start += (start < 0) ? N : 0;
        start = (start < 0) ? ((step < 0) ? -1 : 0) : start;
        start = (start >= N) ? ((step < 0) ? N - 1 : N) : start;
        int seq_cnt = 0;
        if (step > 0 && stop >= start) {
            seq_cnt = (stop - start + step - 1) / step;
        } else if (step < 0 && stop <= start) {
            seq_cnt = (stop - start + step + 1) / step;
        }
        indices_len = static_cast<size_t>(seq_cnt);
    }

    int numnew, index = 0;
    mxnet::TShape val_newshape(arr.shape_.ndim(), -1);
    for (int i = inputs[insert_::kValues].shape_.ndim() - 1, j = arr.shape_.ndim() - 1;
         i >= 0 || j >= 0; --i, --j) {
        if (i >= 0 && j >= 0) {
            val_newshape[j] = inputs[insert_::kValues].shape_[i];
        } else if (i >= 0) {
            CHECK_EQ(inputs[insert_::kValues].shape_[i], 1) << "index exceed limits.";
        } else {
            val_newshape[j] = 1;
        }
    }
    values = inputs[insert_::kValues].reshape(val_newshape);

    mxnet::TShape old_valshape(values.shape_);
    if (param.int_ind.has_value() ||
      (inputs.size() == 3U && inputs[insert_::kObj].shape_.ndim() == 0)) {
        if (param.int_ind.has_value()) {
          index = param.int_ind.value();
          CHECK(index >= -1 * N && index <= N)
            << "Index should be in the range of [-r, r-1] where r is the dim size in 'axis'";
          if (index < 0) {
            index += N;
          }
        }
        numnew = values.shape_[0];

        // If 'obj' is a int, then, values = moveaxis(values, 0, axis)
        mxnet::TShape axes(values.ndim(), -1);
        mxnet::TShape val_newshape(values.ndim(), -1);
        int axes_id = 0;
        for (int i = 1; i <= axis; ++i) {
            axes[axes_id++] = i;
        }
        axes[axes_id++] = 0;
        for (int i = axis + 1; i < values.ndim(); ++i) {
            axes[axes_id++] = i;
        }
        for (int i = 0; i < values.ndim(); ++i) {
            val_newshape[i] = values.shape_[axes[i]];
        }
        values.shape_.assign(val_newshape.begin(), val_newshape.end());
        newshape[axis] += numnew;
    } else if (indices_len == 1) {
        numnew = values.shape_[axis];
        newshape[axis] += numnew;
        if (param.start.has_value()) {
            index = start;
            CHECK(index >= -1 * N && index <= N)
                << "Index should be in the range of [-r, r-1] where r is the dim size in 'axis'";
            if (index < 0) {
                index += N;
            }
        }
    } else {
      numnew = static_cast<int>(indices_len);
      newshape[axis] += numnew;
    }

    const mxnet::TShape& outshape = outputs[insert_::kOut].shape_;
    mshadow::Shape<10> arr_strides;
    int stride = 1;
    for (int i = arr.shape_.ndim() - 1; i >= 0; --i) {
        arr_strides[i] = stride;
        stride *= arr.shape_[i];
    }
    mshadow::Shape<10> val_strides;
    stride = 1;
    for (int i = values.shape_.ndim() - 1; i >= 0; --i) {
        val_strides[i] = stride;
        stride *= values.shape_[i];
    }
    mshadow::Shape<10> old_val_strides;
    stride = 1;
    for (int i = old_valshape.ndim() - 1; i >= 0; --i) {
        old_val_strides[i] = stride;
        stride *= old_valshape[i];
    }
    mshadow::Shape<10> out_strides;
    stride = 1;
    for (int i = outshape.ndim() - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= outshape[i];
    }
    mshadow::Shape<10> k_outshape;
    for (int i = 0 ; i < outshape.ndim() ; ++i) {
        k_outshape[i] = outshape[i];
    }
    mshadow::Shape<10> k_valshape;
    for (int i = 0 ; i < values.shape_.ndim() ; ++i) {
        k_valshape[i] = values.shape_[i];
    }

    if (param.int_ind.has_value()) {
      MSHADOW_TYPE_SWITCH(outputs[insert_::kOut].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[insert_::kOut], req_type, {
          Kernel<InsertSingleIndexForward<req_type>, xpu>::Launch(s, outshape.Size(),
                                            outputs[insert_::kOut].dptr<DType>(),
                                            values.dptr<DType>(), arr.dptr<DType>(),
                                            k_outshape, k_valshape, index, numnew,
                                            val_strides, old_val_strides, arr_strides,
                                            out_strides, arr.shape_.ndim(),
                                            values.shape_.ndim(), outshape.ndim(),
                                            axis, true);
        });
      });
    } else if (inputs.size() == 3U && inputs[insert_::kObj].shape_.ndim() == 0) {
      MSHADOW_TYPE_SWITCH(outputs[insert_::kOut].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[insert_::kOut], req_type, {
          MSHADOW_TYPE_SWITCH(inputs[insert_::kObj].type_flag_, IType, {
            Kernel<InsertSingleIndexForward<req_type>, xpu>::Launch(s, outshape.Size(),
                                            outputs[insert_::kOut].dptr<DType>(),
                                            values.dptr<DType>(), arr.dptr<DType>(),
                                            k_outshape, k_valshape, N,
                                            inputs[insert_::kObj].dptr<IType>(), numnew,
                                            val_strides, old_val_strides, arr_strides,
                                            out_strides, arr.shape_.ndim(),
                                            values.shape_.ndim(), outshape.ndim(),
                                            axis, true);
          });
        });
      });
    } else if (indices_len == 1) {
      MSHADOW_TYPE_SWITCH(outputs[insert_::kOut].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[insert_::kOut], req_type, {
          if (param.stop.has_value()) {
            Kernel<InsertSingleIndexForward<req_type>, xpu>::Launch(s, outshape.Size(),
                                            outputs[insert_::kOut].dptr<DType>(),
                                            values.dptr<DType>(), arr.dptr<DType>(),
                                            k_outshape, k_valshape, start, numnew,
                                            val_strides, old_val_strides, arr_strides, out_strides,
                                            arr.shape_.ndim(), values.shape_.ndim(),
                                            outshape.ndim(), axis, false);
          } else {
            MSHADOW_TYPE_SWITCH(inputs[insert_::kObj].type_flag_, IType, {
              Kernel<InsertSingleIndexForward<req_type>, xpu>::Launch(s, outshape.Size(),
                                                outputs[insert_::kOut].dptr<DType>(),
                                                values.dptr<DType>(), arr.dptr<DType>(),
                                                k_outshape, k_valshape,
                                                N, inputs[insert_::kObj].dptr<IType>(), numnew,
                                                val_strides, old_val_strides,
                                                arr_strides, out_strides,
                                                arr.shape_.ndim(), values.shape_.ndim(),
                                                outshape.ndim(), axis, false);
            });
          }
        });
      });
    } else {
        // broadcast check
        for (int i = outshape.ndim() - 1; i >= 0; --i) {
            int sz = outshape[i];
            if (i == axis) {
                sz = numnew;
            }
            CHECK((values.shape_[i] == 1) || (values.shape_[i] == sz));
        }
        size_t temp_storage_bytes, temp_mem_size;
        MSHADOW_TYPE_SWITCH((inputs.size() == 3U) ?
                             inputs[insert_::kObj].type_flag_ :
                             mshadow::DataType<int64_t>::kFlag, IType, {
          temp_storage_bytes = SortByKeyWorkspaceSize<IType, int, xpu>(indices_len, false, true);
          temp_mem_size = indices_len * sizeof(IType) * 2 +
                                      indices_len * sizeof(int) +
                                      newshape[axis] * sizeof(int) * 2 +
                                      temp_storage_bytes;
          Tensor<xpu, 1, char> temp_mem =
                  ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
          IType* indices_ptr = reinterpret_cast<IType*>(temp_mem.dptr_);
          IType* sorted_indices_ptr = reinterpret_cast<IType*>
                                      (temp_mem.dptr_ + indices_len * sizeof(IType));
          int* order_ptr = reinterpret_cast<int*>(temp_mem.dptr_ + indices_len * sizeof(IType) * 2);
          int* is_insert = reinterpret_cast<int*>(temp_mem.dptr_ + indices_len * sizeof(IType) * 2
                            + indices_len * sizeof(int));
          int* origin_idx = reinterpret_cast<int*>(temp_mem.dptr_ +  indices_len * sizeof(IType) * 2
                            + indices_len * sizeof(int) + newshape[axis] * sizeof(int));
          Tensor<xpu, 1, char> temp_storage(temp_mem.dptr_ +  indices_len * sizeof(IType) * 2
                            + indices_len * sizeof(int) + newshape[axis] * sizeof(int) * 2,
                            Shape1(temp_storage_bytes), s);
          Tensor<xpu, 1, IType> indices(indices_ptr, Shape1(indices_len), s);
          Tensor<xpu, 1, IType> sorted_indices(sorted_indices_ptr, Shape1(indices_len), s);
          Tensor<xpu, 1, int> order(order_ptr, Shape1(indices_len), s);
          int num_bits = common::ilog2ui(static_cast<unsigned int>(indices_len) - 1);

          if (param.stop.has_value()) {
            Kernel<SliceToIndices, xpu>::Launch(s, indices_len,
                                        indices_ptr, N,
                                        start, step);
          } else {
            Kernel<ObjToIndices, xpu>::Launch(s, indices_len,
                                      indices_ptr, N,
                                      inputs[insert_::kObj].dptr<IType>());
          }

          Kernel<AssignId, xpu>::Launch(s, indices_len, order_ptr);
          mxnet::op::SortByKey(indices, order, true, &temp_storage, 0, num_bits, &sorted_indices);
          Kernel<IndicesModify, xpu>::Launch(s, indices_len, indices_ptr, order_ptr);

          Kernel<AssignInsertZero, xpu>::Launch(s, newshape[axis], is_insert);
          Kernel<SetIsInsert, xpu>::Launch(s, indices_len, indices_ptr, is_insert);

          Kernel<SetOriginValuesIdx, xpu>::Launch(s, indices_len, indices_ptr, origin_idx);
          Kernel<SetOriginArrIdx, xpu>::Launch(s, newshape[axis], is_insert, origin_idx);

          MSHADOW_TYPE_SWITCH(outputs[insert_::kOut].type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[insert_::kOut], req_type, {
              Kernel<InsertSeqForward<req_type>, xpu>::Launch(s, outshape.Size(),
                                              outputs[insert_::kOut].dptr<DType>(),
                                              values.dptr<DType>(), arr.dptr<DType>(),
                                              k_outshape, k_valshape, is_insert, origin_idx,
                                              val_strides, arr_strides, out_strides,
                                              arr.shape_.ndim(), values.shape_.ndim(),
                                              outshape.ndim(), axis);
            });
          });
        });
    }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_
