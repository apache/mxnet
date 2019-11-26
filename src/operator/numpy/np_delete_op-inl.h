/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_delete_op-inl.h
 * \brief Function definition of delete operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_

#include <vector>
#include <memory>
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

namespace mxnet {
namespace op {

struct NumpyDeleteParam : public dmlc::Parameter<NumpyDeleteParam> {
  dmlc::optional<int> start;
  dmlc::optional<int> stop;
  dmlc::optional<int> step;
  dmlc::optional<int> int_ind;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyDeleteParam) {
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

namespace delete_ {
enum DeleteOpInputs {kArr, kObj};
enum DeleteOpOutputs {kOut};
}  // namespace delete_

template<int req>
struct Copy {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

struct SliceToIndices {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, int start, int step) {
    indices[i] = start + i * step;
  }
};

struct ObjToIndices {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, const IType* obj) {
    indices[i] = obj[i];
  }
};

template<typename IType>
struct AssignNum {
  MSHADOW_XINLINE static void Map(int i, IType* output, const IType data) {
    output[i] = data;
  }
};

struct IsDeleteCal {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, int N, bool* is_delete, const IType* indices) {
    if ((static_cast<int64_t>(indices[i]) >= 0) &&
      (static_cast<int64_t>(indices[i]) < N)) {
      is_delete[static_cast<int64_t>(indices[i])] = true;
    }
  }
};

struct OutPosCal {
  /*!
   * \brief map the index from input to output
   */
  MSHADOW_XINLINE static void Map(int i, int64_t* out_pos, const bool* is_delete) {
    if (!is_delete[i]) {
      int cnt = 0;
      for ( int j = 0; j < i; ++j) {
        if (!is_delete[j]) {
          cnt++;
        }
      }
      out_pos[i] = cnt;
    }
  }
};

template<int ndim>
inline mshadow::Shape<ndim> GetStride(const mxnet::TShape& shape) {
  mshadow::Shape<ndim>stride;
  size_t tmp = 1;
  for (int i = shape.ndim() - 1; i >= 0; --i) {
    stride[i] = tmp;
    tmp *= shape[i];
  }
  return stride;
}

template<int ndim>
inline mshadow::Shape<ndim> GetKernelShape(const mxnet::TShape& shape) {
  mshadow::Shape<ndim>k_shape;
  for (int i = 0 ; i < shape.ndim() ; ++i) {
    k_shape[i] = shape[i];
  }
  return k_shape;
}

template<int req>
struct DeleteImpl {
  /*!
   * \brief delete a sub-array from input along an axis according to 'is_delete'.
   * \tparam xpu - cpu or gpu.
   * \param out_data - output: a new array with sub-arrays along an axis deleted.
   * \param in_arr - input: 'arr', original array.
   * \param is_delete - mark where will be deleted or be reminded in 'arr'
   * \param out_pos - if is_delete[i] is 'false', out_pos[i] indicates its.
   * \param arrshape - the shape of 'arr'.
   * \param arr_stride - the stride of 'arr'.
   * \param out_stride - the stride of 'out_data'.
   * \param out_ndim - the ndim of 'out_data'.
   * \param axis - delete sub-array along this axis
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const DType* in_arr,
                                  const bool* is_delete,
                                  const int64_t* out_pos,
                                  const mshadow::Shape<10> arrshape,
                                  const mshadow::Shape<10> arr_stride,
                                  const mshadow::Shape<10> out_stride,
                                  const int out_ndim, const int axis) {
    const int64_t arr_head = i / arr_stride[axis];
    const int64_t arr_mid = arr_head % arrshape[axis];
    mshadow::Shape<10> arr_idx;  // i -> position in in_arr's shape
    for (int j = 0; j < out_ndim; ++j) {
      const int64_t head = i / arr_stride[j];
      const int64_t mid = head % arrshape[j];
      arr_idx[j] = mid;
    }
    if (!is_delete[arr_mid]) {
      arr_idx[axis] = out_pos[arr_mid];
      int64_t dest_idx = 0;
      for (int j =0; j < out_ndim; ++j) {
        dest_idx += out_stride[j] * arr_idx[j];
      }
      KERNEL_ASSIGN(out_data[dest_idx], req, in_arr[i]);
    }
  }
};

template<typename xpu>
void NumpyDeleteCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  const NumpyDeleteParam& param = nnvm::get<NumpyDeleteParam>(attrs.parsed);
  CHECK_EQ(inputs.size(),
          (param.step.has_value() || param.int_ind.has_value()) ? 1U : 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  int ndim = inputs[delete_::kArr].shape().ndim();
  int axis = param.axis.has_value() ? param.axis.value() : -1;
  NDArray arr;

  if (!param.axis.has_value()) {
    arr = inputs[delete_::kArr].Reshape(Shape1(inputs[delete_::kArr].shape().Size()));
    ndim = 1;
    axis = -1;
  } else {
    arr = inputs[delete_::kArr];
  }

  if (ndim == 0) {
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(arr.shape());
    MSHADOW_TYPE_SWITCH(outputs[delete_::kOut].dtype(), DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[delete_::kOut], req_type, {
        Kernel<Copy<req_type>, xpu>::Launch(
            s, outputs[delete_::kOut].shape().Size(),
            outputs[delete_::kOut].data().dptr<DType>(),
            inputs[delete_::kArr].data().dptr<DType>());
      });
    });
    return;
  }

  axis = CheckAxis(axis, ndim);
  int N = (arr.shape())[axis];
  mxnet::TShape newshape(arr.shape());
  int start = 0, stop = 0, step = 0;
  size_t numtodel = 0;
  int index = 0;

  if (param.step.has_value()) {
    step = param.step.value();
    CHECK_NE(step, 0) << "'step' can not equal to 0.";
    if (param.stop.has_value()) {
      stop = param.stop.value();
      stop += (stop < 0) ? N : 0;
      stop = (stop < 0) ? ((step < 0) ? -1 : 0) : stop;
      stop = (stop >= N) ? ((step < 0) ? N - 1 : N) : stop;
    } else {
      stop = (step > 0) ? N : -1;
    }
    if (param.start.has_value()) {
      start = param.start.value();
      start += (start < 0) ? N : 0;
      start = (start < 0) ? ((step < 0) ? -1 : 0) : start;
      start = (start >= N) ? ((step < 0) ? N - 1 : N) : start;
    } else {
      start = (step > 0) ? 0 : N - 1;
    }
    if (step > 0 && stop >= start) {
      numtodel = static_cast<size_t>((stop - start + step - 1) / step);
    } else if (step < 0 && stop <= start) {
      numtodel = static_cast<size_t>((stop - start + step + 1) / step);
    }
    if (numtodel == 0) {
      const_cast<NDArray &>(outputs[delete_::kOut]).Init(arr.shape());
      MSHADOW_TYPE_SWITCH(outputs[delete_::kOut].dtype(), DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[delete_::kOut], req_type, {
          Kernel<Copy<req_type>, xpu>::Launch(
                s, outputs[delete_::kOut].shape().Size(),
                outputs[delete_::kOut].data().dptr<DType>(),
                inputs[delete_::kArr].data().dptr<DType>());
        });
      });
      return;
    }
    newshape[axis] -= numtodel;
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(newshape);
  } else if (param.int_ind.has_value()) {
    index = param.int_ind.value();
    CHECK((index >= -1 * N) && (index < N))
      << "index " << index
      << " is out of bounds for axis " << axis
      << " with size " << N << "\n";
    index += ((index < 0) ? N : 0);
    numtodel = static_cast<size_t>(1);
    newshape[axis] -= 1;
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(newshape);
  } else {
    numtodel = inputs[delete_::kObj].shape().Size();
  }

  MSHADOW_TYPE_SWITCH((inputs.size() == 2U) ?
                       inputs[delete_::kObj].dtype() :
                       mshadow::DataType<int64_t>::kFlag, IType, {
    size_t temp_mem_size = sizeof(int64_t) * arr.shape()[axis] +
                           sizeof(IType) * numtodel +
                           sizeof(bool) * arr.shape()[axis];
    Tensor<xpu, 1, char> temp_mem =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    int64_t* out_pos_ptr = reinterpret_cast<int64_t*>(temp_mem.dptr_);
    IType* indices_ptr = reinterpret_cast<IType*>
                         (temp_mem.dptr_ + sizeof(int64_t) * arr.shape()[axis]);
    bool* is_delete_ptr = reinterpret_cast<bool*>
                          (temp_mem.dptr_ + sizeof(int64_t) * arr.shape()[axis]
                          + sizeof(IType) * numtodel);
    if (param.step.has_value()) {
      Kernel<SliceToIndices, xpu>::Launch(s, numtodel,
                                          indices_ptr, start, step);
    } else if (param.int_ind.has_value()) {
      Kernel<AssignNum<IType>, xpu>::Launch(s, numtodel, indices_ptr, index);
    } else {
      Kernel<ObjToIndices, xpu>::Launch(s, numtodel, indices_ptr,
                                        inputs[delete_::kObj].data().dptr<IType>());
    }
    Kernel<AssignNum<bool>, xpu>::Launch(s, arr.shape()[axis], is_delete_ptr, false);
    Kernel<IsDeleteCal, xpu>::Launch(s, numtodel, N, is_delete_ptr, indices_ptr);
    Kernel<OutPosCal, xpu>::Launch(s, arr.shape()[axis], out_pos_ptr, is_delete_ptr);
    if (inputs.size() == 2U) {
      IType* input_obj = inputs[delete_::kObj].data().dptr<IType>();
      #ifndef __CUDACC__
        std::vector<bool>vec_is_delete(is_delete_ptr, is_delete_ptr + arr.shape()[axis]);
      #else
        thrust::device_ptr<bool>is_delete_dev(is_delete_ptr);
        thrust::device_vector<bool>vec_is_delete(is_delete_dev, is_delete_dev + arr.shape()[axis]);
      #endif
      numtodel = 0;
      for (int i = 0; i < arr.shape()[axis]; ++i) {
        if (vec_is_delete[i]) {
          numtodel++;
        }
      }
      newshape[axis] -= numtodel;
      const_cast<NDArray &>(outputs[delete_::kOut]).Init(newshape);
    }
    mshadow::Shape<10> arr_strides = GetStride<10>(arr.shape());
    mshadow::Shape<10> out_strides = GetStride<10>(newshape);
    mshadow::Shape<10> k_arrshape = GetKernelShape<10>(arr.shape());
    MSHADOW_TYPE_SWITCH(outputs[delete_::kOut].dtype(), DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[delete_::kOut], req_type, {
        Kernel<DeleteImpl<req_type>, xpu>::Launch(s, arr.shape().Size(),
                                                  outputs[delete_::kOut].data().dptr<DType>(),
                                                  arr.data().dptr<DType>(),
                                                  is_delete_ptr, out_pos_ptr,
                                                  k_arrshape,
                                                  arr_strides, out_strides,
                                                  newshape.ndim(), axis);
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_
