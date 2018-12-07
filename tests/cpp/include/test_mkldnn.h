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
 *  \file test_mkldnn.h
 *  \brief helper functions to test mkldnn.
 *  \author Alex Zai
 */

#ifndef TEST_MKLDNN_H_
#define TEST_MKLDNN_H_

#if MXNET_USE_MKLDNN == 1

#include <set>
#include <string>
#include <vector>
#include "../../../3rdparty/mkldnn/include/mkldnn_types.h"
#include "../../../3rdparty/googletest/googletest/include/gtest/gtest.h"
#include "../../../src/operator/nn/mkldnn/mkldnn_base-inl.h"

using namespace mxnet;

inline static mkldnn::memory::primitive_desc GetMemPD(const TShape s, int dtype,
                                               mkldnn::memory::format format) {
  mkldnn::memory::dims dims(s.ndim());
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = s[i];
  mkldnn::memory::desc desc{dims, get_mkldnn_type(dtype), format};
  return mkldnn::memory::primitive_desc(desc, CpuEngine::Get()->get_engine());
}

inline static mkldnn::memory::primitive_desc GetExpandedMemPD(
    mkldnn::memory::primitive_desc pd, float scale, int dim = 0) {
  CHECK(dim < pd.desc().data.ndims) << "dimension cannot be larger than total dimensions of input";
  nnvm::TShape s(pd.desc().data.ndims);
  for (size_t i = 0; i < pd.desc().data.ndims; i++)
    s[i] = pd.desc().data.dims[i];
  s[dim] = static_cast<int>(s[dim] * scale);
  return GetMemPD(s, mshadow::DataType<mshadow::default_real_t>::kFlag,
                  static_cast<mkldnn::memory::format>(pd.desc().data.format));
}

struct TestArrayShapes {
  std::vector<nnvm::TShape> shapes;
  std::vector<mkldnn::memory::primitive_desc> pds;
};

// Init arrays with the default layout.
inline static void InitDefaultArray(NDArray *arr, bool is_rand = false) {
  const TBlob &blob = arr->data();
  mshadow::default_real_t *data = blob.dptr<mshadow::default_real_t>();
  int size = blob.Size();

  for (int i = 0; i < size; i++)
    if (is_rand) {
      data[i] = (std::rand() % 100) - 50;
    } else {
      data[i] = i % 100 - 50;
    }
}


// Init arrays with the specified layout.
inline static void InitMKLDNNArray(NDArray *arr, const mkldnn::memory::primitive_desc &pd,
                            bool is_rand = false) {
  InitDefaultArray(arr, is_rand);
  arr->MKLDNNDataReorderAsync(pd);
  arr->WaitToRead();
}

inline static bool IsSameShape(mkldnn::memory::primitive_desc pd, TShape shape) {
  if (pd.desc().data.ndims != shape.ndim()) return false;
  for (size_t i = 0; i < shape.ndim(); i++)
    if (pd.desc().data.dims[i] != shape[i]) return false;
  return true;
}

// This function gets special MKLDNN formats without knowing the specific
// hardware configuration. Certainly, it potentially misses some format if
// it's specific for certain array shapes. It covers at least one special format
// for each of the formats: nchw, oihw, goihw.
// To test the logic of the code in NDArray, these formats should be enough.
inline static std::vector<mkldnn::memory::format> GetMKLDNNFormat(size_t num_dims, int dtype) {
  if (num_dims == 4) {
    mkldnn::memory::dims data_dims{1, 3, 224, 224};
    mkldnn::memory::desc data_md{data_dims, get_mkldnn_type(dtype),
                                 mkldnn::memory::format::any};
    mkldnn::memory::dims weight_dims{96, 3, 11, 11};
    mkldnn::memory::desc weight_md{weight_dims, get_mkldnn_type(dtype),
                                   mkldnn::memory::format::any};
    mkldnn::memory::dims output_dims{1, 96, 54, 54};
    mkldnn::memory::desc out_md{output_dims, get_mkldnn_type(dtype),
                                mkldnn::memory::format::any};
    mkldnn::memory::dims strides{4, 4};
    mkldnn::memory::dims padding{0, 0};

    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
                                           mkldnn::algorithm::convolution_direct,
                                           data_md, weight_md, out_md, strides,
                                           padding, padding, mkldnn::padding_kind::zero);
    mkldnn::convolution_forward::primitive_desc pd(desc, CpuEngine::Get()->get_engine());
    while (pd.dst_primitive_desc().get_size() != GetMemDescSize(out_md) ||
           pd.src_primitive_desc().get_size() != GetMemDescSize(data_md) ||
           pd.weights_primitive_desc().get_size() != GetMemDescSize(weight_md)) {
      CHECK(pd.next_impl()) << "No implementation";
    }

    std::vector<mkldnn::memory::format> ret(1);
    ret[0] = static_cast<mkldnn::memory::format>(pd.dst_primitive_desc().desc().data.format);
    printf("format: %d \n", ret[0]);
    return ret;
  } else if (num_dims == 5) {
    mkldnn::memory::dims data_dims{1, 32, 112, 112};
    mkldnn::memory::desc data_md{data_dims, get_mkldnn_type(dtype),
                                 mkldnn::memory::format::any};
    mkldnn::memory::dims weight_dims{32, 1, 1, 3, 3};
    mkldnn::memory::desc weight_md{weight_dims, get_mkldnn_type(dtype),
                                   mkldnn::memory::format::any};
    mkldnn::memory::dims output_dims{1, 32, 112, 112};
    mkldnn::memory::desc out_md{output_dims, get_mkldnn_type(dtype),
                                mkldnn::memory::format::any};
    mkldnn::memory::dims strides{1, 1};
    mkldnn::memory::dims padding{1, 1};

    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
                                           mkldnn::algorithm::convolution_direct,
                                           data_md, weight_md, out_md, strides,
                                           padding, padding, mkldnn::padding_kind::zero);
    mkldnn::convolution_forward::primitive_desc pd(desc, CpuEngine::Get()->get_engine());
    while (pd.dst_primitive_desc().get_size() != GetMemDescSize(out_md) ||
           pd.src_primitive_desc().get_size() != GetMemDescSize(data_md) ||
           pd.weights_primitive_desc().get_size() != GetMemDescSize(weight_md)) {
      CHECK(pd.next_impl()) << "No implementation";
    }

    std::vector<mkldnn::memory::format> ret(1);
    ret[0] = static_cast<mkldnn::memory::format>(pd.weights_primitive_desc().desc().data.format);
    printf("format: %d\n", ret[0]);
    return ret;
  } else {
    return std::vector<mkldnn::memory::format>();
  }
}

inline static TestArrayShapes GetTestArrayShapes(bool spatial_data_format = false) {
  int dtype = mshadow::DataType<mshadow::default_real_t>::kFlag;
  std::vector<TShape> shapes;
  std::vector<mkldnn::memory::primitive_desc> pds;
  {
    // 1D
    TShape s(1);
    s[0] = 279936;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::x));
    s[0] = 34848;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::x));
  }
  {
    // 2D
    TShape s(2);
    s[0] = 96;
    s[1] = 2916;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::nc));
    s[0] = 96;
    s[1] = 363;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::nc));
  }
  {
    // 4D
    TShape s1(4);
    s1[0] = 10; s1[1] = 96; s1[2] = 54; s1[3] = 54;
    shapes.push_back(s1);
    pds.push_back(GetMemPD(s1, dtype, mkldnn::memory::format::nchw));

    TShape s2(4);
    s2[0] = 96; s2[1] = 3; s2[2] = 11; s2[3] = 11;
    shapes.push_back(s2);
    pds.push_back(GetMemPD(s2, dtype, mkldnn::memory::format::oihw));

    std::vector<mkldnn::memory::format> formats = GetMKLDNNFormat(4, dtype);
    if (!spatial_data_format) {
      pds.push_back(GetMemPD(s1, dtype, formats[0]));
    }
  }
  {
    // 5D
    TShape s(5);
    s[0] = 96; s[1] = 1; s[2] = 3; s[3] = 11; s[4] = 11;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::goihw));

    std::vector<mkldnn::memory::format> formats = GetMKLDNNFormat(5, dtype);
    if (!spatial_data_format) {
      pds.push_back(GetMemPD(s, dtype, formats[0]));
    }
  }

  TestArrayShapes ret;
  ret.shapes = shapes;
  ret.pds = pds;
  return ret;
}

struct NDArrayAttrs {
  NDArray arr;
  std::string desc;
  NDArrayAttrs(NDArray arr, std::string desc) : arr(arr), desc(desc) {}
};

struct OpAttrs {
  nnvm::NodeAttrs attrs;
  std::vector<DispatchMode> dispatches;
  std::set<OpReqType> requests;
  std::unordered_set<int> accept_dims;
  int num_inputs;
  int num_outputs;
  int input_types;
  int output_types;
};

enum ArrayTypes {
  Normal = 1,
  MKLDNN = 2,
  MKLDNNDiffShape = 4,
  MKLDNNDiffDim = 8,
  NormalReshaped = 16,
  MKLDNNReshaped = 32,
  MKLDNNReshapedDiffShape = 64,
  MKLDNNReshapedDiffDim = 128,
  NormalReused = 256,
  MKLDNNReused = 512,
  MKLDNNReusedDiffDim = 1024,
  NormalReshapedReused = 2048,
  NormalReusedDiffDtype = 4096,
  All = 8191,
};


inline NDArray CreateKernelNDArray(TShape kernel, int num_filters, TShape input,
    bool is_deconv = false) {
  CHECK_EQ(kernel.ndim(), 2) << "mkldnn only supports 2d filters on 4d inputs";
  TShape target_shape(4);
  target_shape[0] = is_deconv ? input[1] : num_filters;
  target_shape[1] = is_deconv ? num_filters : input[1];
  target_shape[2] = kernel[0];
  target_shape[3] = kernel[1];
  int dtype = mshadow::DataType<mshadow::default_real_t>::kFlag;
  NDArray arr(target_shape, Context());
  auto pd = GetMemPD(target_shape, dtype, mkldnn::memory::format::nchw);
  InitMKLDNNArray(&arr, pd);
  return arr;
}

inline NDArray CreateBiasNDArray(TShape target_shape) {
  int dtype = mshadow::DataType<mshadow::default_real_t>::kFlag;
  NDArray arr(target_shape, Context());
  auto pd = GetMemPD(target_shape, dtype, mkldnn::memory::format::x);
  InitMKLDNNArray(&arr, pd);
  return arr;
}

inline int CalculateWidthConvOutput(int width, int kernel, int padding, int stride) {
  return (width - kernel + 2 * padding) / stride  + 1;
}

inline int CalculateWidthDeconvOutput(int width, int kernel, int padding, int stride) {
  return stride * (width - 1) + kernel - 2 * padding;
}

inline std::string CreateShapeString(int value, int dim) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < dim; i++) {
    ss << value;
    if (i != dim - 1) ss << ",";
  }
  ss << ")";
  return ss.str();
}

inline void PrintVerifyMsg(const NDArrayAttrs &arr1, const NDArrayAttrs &arr2) {
  TShape t1 = arr1.arr.shape();
  TShape t2 = arr2.arr.shape();
  std::stringstream ss;
  std::cout << "Verifying: " << arr1.desc.c_str() << " " <<
            t1 << " with " << arr2.desc.c_str() << " " << t2 << "\n";
}

/*
 * We want to get a few types of NDArrays for testing:
 * 1. Normal NDArray
 * 2. Normal NDArray with MKLDNN layout (output from an MKLDNN operator)
 * 3. Normal NDArray with MKLDNN layout whose MKLDNN memory may have different
 *    dimensions from the NDArray (result of MKLDNNDataReorderAsync). However, this
 *    type of NDArrays only exists for weight arrays. I don't think we should
 *    pass them to all operators.
 *    In the inference mode, the MKLDNN memory in the weight array will be
 *    reordered to 5 dimensions.
 * 4. Reshaped/sliced NDArray
 * 5. Reshaped/sliced NDArray with MKLDNN layout (reshape/slice from Normal NDArray
 *    with MKLDNN layout)
 * 6. Reshaped/sliced NDArray with MKLDNN layout whose MKLDNN memory may have
 *    different dimensions from the NDArray (result of MKLDNNDataReorderAsync).
 *    However, this type of NDArrays only exists for weight arrays. I don't think
 *    we should pass them to all operators.
 *    In the inference mode, the MKLDNN memory in the weight array will be
 *    reordered to 5 dimensions.
 *
 *  num_inputs / dim arguments used to scale shape (used for concat backwards to enlarge input shapes)
 */
inline std::vector<NDArrayAttrs> GetTestInputArrays(
    int types = ArrayTypes::All, bool rand = false,
    std::vector<float> scale = {1}, bool spatial_data_format = false) {
  TestArrayShapes tas = GetTestArrayShapes(spatial_data_format);
  std::vector<nnvm::TShape> shapes = tas.shapes;
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  std::vector<NDArrayAttrs> in_arrs;
  std::string desc;

  int slice_amount = scale[0];
  for (auto shape : shapes) {
    if (scale.size() > shape.ndim())
      continue;

    for (size_t dim = 0; dim < scale.size(); ++dim)
      shape[dim] = static_cast<int>(round(shape[dim] * scale[dim]));

    // Type 1.
    NDArray arr(shape, Context());
    if (types & ArrayTypes::Normal) {
      InitDefaultArray(&arr, rand);
      in_arrs.emplace_back(arr, "Normal NDArray");
    }

    // Type 4
    arr = NDArray(shape, Context());
    if (types & ArrayTypes::NormalReshaped) {
      InitDefaultArray(&arr, rand);
      in_arrs.emplace_back(arr.Slice(slice_amount, arr.shape()[0] - slice_amount),
                           "Reshaped Normal NDArray");
    }


    for (auto pd : pds) {
      for (size_t dim = 0; dim < scale.size(); ++dim) {
        // preserve if matching layout else just expand on 0 dim
        if (shape.ndim() == pd.desc().data.ndims)
          pd = GetExpandedMemPD(pd, scale[dim], dim);
        else
          pd = GetExpandedMemPD(pd, scale[dim]);
      }

      if (shape.Size() != pd.get_size() / sizeof(mshadow::default_real_t))
        continue;

      // Type 2, 3.
      arr = NDArray(shape, Context());
      if (shape.ndim() == pd.desc().data.ndims && IsSameShape(pd, shape)
          && types & ArrayTypes::MKLDNN) {
        desc = "MKLDNN NDArray";
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr, desc);
      } else if (shape.ndim() == pd.desc().data.ndims && !IsSameShape(pd, shape)
          && types & ArrayTypes::MKLDNNDiffShape) {
        desc = "MKLDNN NDArray with different shape";
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr, desc);
      } else if (shape.ndim() != pd.desc().data.ndims && types & ArrayTypes::MKLDNNDiffDim) {
        std::stringstream ss;
        ss << "MKLDNN NDArray with different dim " <<
           shape.ndim() << "/" << pd.desc().data.ndims;
        desc = ss.str();
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr, desc);
      }


      // Type 5, 6.
      arr = NDArray(shape, Context());
      if (shape.ndim() == pd.desc().data.ndims && IsSameShape(pd, shape)
          && types & ArrayTypes::MKLDNNReshaped) {
        desc = "Reshaped MKLDNN NDArray";
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr.Slice(slice_amount, arr.shape()[0] - slice_amount), desc);
      } else if (shape.ndim() == pd.desc().data.ndims && !IsSameShape(pd, shape)
          && types & ArrayTypes::MKLDNNReshapedDiffShape) {
        desc = "Reshaped MKLDNN NDArray with different shape";
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr.Slice(slice_amount, arr.shape()[0] - slice_amount), desc);
      } else if (shape.ndim() != pd.desc().data.ndims
          && types & ArrayTypes::MKLDNNReshapedDiffDim) {
        std::stringstream ss;
        ss << "MKLDNN NDArray with different dim " <<
           shape.ndim() << "/" << pd.desc().data.ndims;
        desc = ss.str();
        InitMKLDNNArray(&arr, pd, rand);
        in_arrs.emplace_back(arr.Slice(slice_amount, arr.shape()[0] - slice_amount), desc);
      }
    }
  }
  return in_arrs;
}

/*
 * We want to get a few types of NDArrays for testing:
 * 1. Normal NDArray
 * 2. Normal NDArray with MKLDNN layout (output from an MKLDNN operator)
 * 3. Normal NDArray with MKLDNN layout whose MKLDNN memory may have different
 *    dimensions from the NDArray (result of MKLDNNDataReorderAsync). However, this
 *    type of NDArrays only exists for weight arrays. I don't think we should
 *    pass them to all operators.
 *    In the inference mode, the MKLDNN memory in the weight array will be
 *    reordered to 5 dimensions.
 * 4. Reshaped/sliced NDArray
 * 5. Reused NDArray (this is created by the MXNet executor). This type of
 *    NDArrays can only be used as output arrays.
 * 6. Reused NDArray converted from an array with a different data type.
 * 7. Reused reshaped/sliced NDArray.
 * 8. Reused NDArray with MKLDNN layout.
 * 9. Reused NDArray with MKLDNN layout of different dimensions.
 *
 * Optional num_inputs / dim args can be passed to modify input shape (used for Concat test)
 */
inline std::vector<NDArrayAttrs> GetTestOutputArrays(
    const TShape &shp,
    const std::vector<mkldnn::memory::primitive_desc> &pds,
    std::vector<float>scale = {1}, bool rand = true, int types = ArrayTypes::All) {
  TShape shape = shp;

  for (int dim = 0; dim < scale.size(); dim++)
    shape[dim] = static_cast<int>(shape[dim] * scale[dim]);

  std::vector<NDArrayAttrs> in_arrs;
  std::string desc;
  // Type 1.
  NDArray arr(shape, Context());

  if (types & ArrayTypes::Normal) {
    in_arrs.emplace_back(arr, "Normal NDArray");
    InitDefaultArray(&in_arrs.back().arr, rand);
  }

  TShape tmp_shape = shape;
  if (types & ArrayTypes::NormalReshaped) {
    // Type 4.
    tmp_shape[0] = shape[0] * 2;
    NDArray arr0(tmp_shape, Context());
    InitDefaultArray(&arr0, rand);
    in_arrs.emplace_back(arr0.Slice(1, shape[0] + 1), "Reshaped NDArray");
  }

  nnvm::TShape s(1);
  if (types & ArrayTypes::NormalReused) {
    // Type 5.
    // Get a reused version.
    s[0] = shape.Size();
    NDArray arr1(s, Context());
    arr1 = arr1.AsArray(shape, arr1.dtype());
    InitDefaultArray(&arr1, rand);
    in_arrs.emplace_back(arr1, "Reused NDArray");
  }

  if (types & ArrayTypes::NormalReusedDiffDtype) {
    // Type 6.
    s[0] = shape.Size() * GetTypeSize(mshadow::default_type_flag);
    NDArray arr2(s, Context(), true, mshadow::kUint8);
    arr2 = arr2.AsArray(shape, mshadow::default_type_flag);
    InitDefaultArray(&arr2, rand);
    in_arrs.emplace_back(arr2, "Reused NDArray with diff data type");
  }

  if (types & ArrayTypes::NormalReshapedReused) {
    // Type 7
    s[0] = shape.Size() * GetTypeSize(mshadow::default_type_flag) * 2;
    NDArray arr3(s, Context(), true, mshadow::kUint8);
    tmp_shape[0] = shape[0] * 2;
    arr3 = arr3.AsArray(tmp_shape, mshadow::default_type_flag);
    InitDefaultArray(&arr3, rand);
    in_arrs.emplace_back(arr3.Slice(1, shape[0] + 1), "Reused+Reshaped NDArray");
  }

  for (auto pd : pds) {
    if (shape.Size() != pd.get_size() / sizeof(mshadow::default_real_t))
      continue;

    if (scale.size() > pd.desc().data.ndims)
      continue;

    for (int dim = 0; dim < scale.size(); dim++)
      pd = GetExpandedMemPD(pd, scale[dim]);

    // Type 2, 3.
    arr = NDArray(shape, Context());
    desc = "MKLDNN NDArray";
    if (shape.ndim() != pd.desc().data.ndims) {
      std::stringstream ss;
      ss << "MKLDNN NDArray with different memory layout "
         << shape.ndim() << "/" << pd.desc().data.ndims;
      desc = ss.str();
    }

    if ((types & ArrayTypes::MKLDNN && shape.ndim() == pd.desc().data.ndims) ||
        (types & ArrayTypes::MKLDNNDiffDim && shape.ndim() != pd.desc().data.ndims)) {
      in_arrs.emplace_back(arr, desc);
      InitMKLDNNArray(&in_arrs.back().arr, pd, rand);
    }

    // Type 8, 9.
    // Get a reused version.
    nnvm::TShape s(1);
    s[0] = shape.Size();
    NDArray arr = NDArray(s, Context());
    arr = arr.AsArray(shape, arr.dtype());
    InitMKLDNNArray(&arr, pd, rand);
    desc = "Reused MKLDNN NDArray";
    if (shape.ndim() != pd.desc().data.ndims) {
      std::stringstream ss;
      ss << "Reused MKLDNN NDArray with different memory layout "
         << shape.ndim() << "/" << pd.desc().data.ndims;
      desc = ss.str();
    }

    if ((types & ArrayTypes::MKLDNNReused && shape.ndim() == pd.desc().data.ndims) ||
        (types & ArrayTypes::MKLDNNReusedDiffDim && shape.ndim() != pd.desc().data.ndims)) {
      in_arrs.emplace_back(arr, desc);
    }
  }
  return in_arrs;
}

/*
 * Determines axis ndarrays are concatenated by
 * Used to verify concat/concat backwards operator
 */
inline int GetDim(TShape input_shape, TShape output_shape) {
  CHECK(input_shape.Size() != output_shape.Size());
  for (size_t i = 0; i < input_shape.ndim(); i++) {
    if (input_shape[i] != output_shape[i])
      return i;
  }
  return -1;
}

/*
 * Calculates the size of continuous block of array inside larger concatenated array
 * Used to verify concat/concat backwards operator
 */
inline int GetBlockSize(TShape shape, int dim) {
  int block_size = 1;
  for (int i = shape.ndim() - 1; i >= dim; i--)
    block_size *= shape[i];
  return block_size;
}

inline int CalculateWidthPoolOutput(int width, int kernel, int padding, int stride) {
  return (width - kernel + 2 * padding) / stride  + 1;
}

using VerifyFunc = std::function<void (const std::vector<NDArray *> &in_arrs,
                                       const std::vector<NDArray *> &out_arrs)>;

inline void VerifyAddRequest(const std::vector<NDArray*> &in_arrs,
                      const std::vector<NDArray*> &original_outputs,
                      const std::vector<NDArray*> &new_outputs,
                      VerifyFunc verify_fn) {
  CHECK(original_outputs.size() == new_outputs.size());
  std::vector<NDArray*> tmp_outputs;
  NDArray tmp;
  for (size_t i = 0; i < new_outputs.size(); i++) {
    tmp = new_outputs[i]->Reorder2Default() - original_outputs[i]->Reorder2Default();
    tmp_outputs.push_back(&tmp);
  }
  Engine::Get()->WaitForAll();
  verify_fn(in_arrs, tmp_outputs);
}

inline void VerifyCopyResult(const std::vector<NDArray *> &in_arrs,
                      const std::vector<NDArray *> &out_arrs) {
  NDArray tmp1 = in_arrs[0]->Reorder2Default();
  NDArray tmp2 = out_arrs[0]->Reorder2Default();
  EXPECT_EQ(tmp1.shape().Size(), tmp2.shape().Size());
  TBlob d1 = tmp1.data();
  TBlob d2 = tmp2.data();
  EXPECT_EQ(memcmp(d1.dptr_, d2.dptr_,
                   tmp1.shape().Size() * sizeof(mshadow::default_real_t)), 0);
}

inline void VerifySumResult(const std::vector<NDArray *> &in_arrs,
                     const std::vector<NDArray *> &out_arrs) {
  NDArray in1 = in_arrs[0]->Reorder2Default();
  NDArray in2 = in_arrs[1]->Reorder2Default();
  NDArray out = out_arrs[0]->Reorder2Default();
  EXPECT_EQ(in1.shape().Size(), in2.shape().Size());
  EXPECT_EQ(in1.shape().Size(), out.shape().Size());

  mshadow::default_real_t *d1 = in1.data().dptr<mshadow::default_real_t>();
  mshadow::default_real_t *d2 = in2.data().dptr<mshadow::default_real_t>();
  mshadow::default_real_t *o = out.data().dptr<mshadow::default_real_t>();
  for (size_t i = 0; i < in1.shape().Size(); i++)
    ASSERT_EQ(d1[i] + d2[i], o[i]);
}

#endif  // MXNET_USE_MKLDNN
#endif  // TEST_MKLDNN_H_
