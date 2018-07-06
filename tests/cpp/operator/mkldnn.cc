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
 *  \file mkldnn.cc
 *  \brief test functions in mkldnn.
 *  \author Da Zheng
 */

#if MXNET_USE_MKLDNN == 1

#include <cmath>
#include <climits>
#include "gtest/gtest.h"
#include "mxnet/imperative.h"
#include "../../src/operator/nn/mkldnn/mkldnn_base-inl.h"
#include "../../src/operator/nn/mkldnn/mkldnn_ops-inl.h"
#include "../../src/operator/nn/pooling-inl.h"

using namespace mxnet;

#if __GNUC__ >= 5
bool test_mem_align(void *mem, size_t size, size_t alignment, size_t space) {
  void *ret1, *ret2;
  size_t space1, space2;
  space1 = space;
  space2 = space;
  ret1 = mxnet::AlignMem(mem, size, alignment, &space1);
  ret2 = std::align(alignment, size, mem, space2);
  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(space1, space2);
  return ret1 == ret2;
}
#endif

TEST(MKLDNN_UTIL_FUNC, AlignMem) {
#if __GNUC__ >= 5
  size_t alignment = 4096;
  void *mem;
  size_t size, space;
  // When mem has been aligned.
  mem = reinterpret_cast<void *>(0x10000);
  size = 1000;
  space = 10000;
  test_mem_align(mem, size, alignment, space);

  // When mem isn't aligned and we have enough space for alignment.
  mem = reinterpret_cast<void *>(0x10010);
  size = 1000;
  space = 10000;
  test_mem_align(mem, size, alignment, space);

  // When mem isn't aligned and we don't have enough memory for alignment
  mem = reinterpret_cast<void *>(0x10010);
  size = 1000;
  space = 1001;
  test_mem_align(mem, size, alignment, space);

  for (size_t i = 0; i < 10000; i++) {
    mem = reinterpret_cast<void *>(random());
    size = random() % 2000;
    space = random() % 2000;
    test_mem_align(mem, size, alignment, space);
  }
#else
  // std::align is not supported in GCC < 5.0, this test case will be checked
  // with newer version
  LOG(INFO) << "Skipped for GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif
}

TEST(MKLDNN_UTIL_FUNC, MemFormat) {
  // Check whether the number of format is correct.
  CHECK_EQ(mkldnn_format_last, 67);
  CHECK_EQ(mkldnn_nchw, 5);
  CHECK_EQ(mkldnn_oihw, 15);
}

// Init arrays with the default layout.
static void InitDefaultArray(NDArray *arr, bool is_rand = false) {
  const TBlob &blob = arr->data();
  mshadow::default_real_t *data = blob.dptr<mshadow::default_real_t>();
  int size = blob.Size();

  for (int i = 0; i < size; i++)
    if (is_rand) {
      data[i] = (std::rand() % 100) - 50;
    } else {
      int shift = size >> 1;
      data[i] = i - shift;
    }
}

using VerifyFunc = std::function<void (const std::vector<NDArray *> &in_arrs,
    const std::vector<NDArray *> &out_arrs)>;

// Init arrays with the specified layout.
static void InitMKLDNNArray(NDArray *arr, const mkldnn::memory::primitive_desc &pd,
                            bool is_rand = false) {
    InitDefaultArray(arr, is_rand);
    arr->MKLDNNDataReorderAsync(pd);
    arr->WaitToRead();
}

static void VerifyDefMem(const mkldnn::memory &mem) {
  mkldnn::memory::primitive_desc pd = mem.get_primitive_desc();
  mshadow::default_real_t *data
      = static_cast<mshadow::default_real_t *>(mem.get_data_handle());
  size_t size = pd.get_size() / sizeof(mshadow::default_real_t);
  size_t num_same = 0;
  int shift = size >> 1;
  for (int i = 0; i < size; i++)
    num_same += data[i] == static_cast<mshadow::default_real_t>(i - shift);
  EXPECT_EQ(num_same, size);
}

static void VerifyMem(const mkldnn::memory &mem) {
  mkldnn::memory::primitive_desc pd = mem.get_primitive_desc();

  if (pd.desc().data.format == GetDefaultFormat(pd.desc())) {
    VerifyDefMem(mem);
  } else {
    mkldnn::memory::dims dims(pd.desc().data.ndims);
    for (size_t i = 0; i < dims.size(); i++)
      dims[i] = pd.desc().data.dims[i];
    mkldnn::memory::desc desc{dims,
                              static_cast<mkldnn::memory::data_type>(pd.desc().data.data_type),
                              static_cast<mkldnn::memory::format>(GetDefaultFormat(pd.desc()))};
    mkldnn::memory::primitive_desc new_pd(desc, CpuEngine::Get()->get_engine());
    mkldnn::memory new_mem(new_pd);

    std::vector<mkldnn::primitive> net;
    net.push_back(mkldnn::reorder(mem, new_mem));
    mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    VerifyDefMem(new_mem);
  }
}

static mkldnn::memory::primitive_desc GetMemPD(const TShape s, int dtype,
                                               mkldnn::memory::format format) {
  mkldnn::memory::dims dims(s.ndim());
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = s[i];
  mkldnn::memory::desc desc{dims, get_mkldnn_type(dtype), format};
  return mkldnn::memory::primitive_desc(desc, CpuEngine::Get()->get_engine());
}

static mkldnn::memory::primitive_desc GetExpandedMemPD(
    mkldnn::memory::primitive_desc pd, float num_input, int dim = 0) {
  CHECK(dim < pd.desc().data.ndims) << "dimension cannot be larger than total dimensions of input";
  nnvm::TShape s(pd.desc().data.ndims);
  for (size_t i = 0; i < pd.desc().data.ndims; i++)
    s[i] = pd.desc().data.dims[i];
  s[dim] = static_cast<int>(s[dim] * num_input);
  return GetMemPD(s, mshadow::DataType<mshadow::default_real_t>::kFlag,
                  static_cast<mkldnn::memory::format>(pd.desc().data.format));
}

// This function gets special MKLDNN formats without knowing the specific
// hardware configuration. Certainly, it potentially misses some format if
// it's specific for certain array shapes. It covers at least one special format
// for each of the formats: nchw, oihw, goihw.
// To test the logic of the code in NDArray, these formats should be enough.
static std::vector<mkldnn::memory::format> GetMKLDNNFormat(size_t num_dims, int dtype) {
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
    std::vector<mkldnn::memory::format> ret(2);
    ret[0] = static_cast<mkldnn::memory::format>(pd.dst_primitive_desc().desc().data.format);
    ret[1] = static_cast<mkldnn::memory::format>(pd.weights_primitive_desc().desc().data.format);
    printf("format: %d, %d\n", ret[0], ret[1]);
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
    std::vector<mkldnn::memory::format> ret(1);
    ret[0] = static_cast<mkldnn::memory::format>(pd.weights_primitive_desc().desc().data.format);
    printf("format: %d\n", ret[0]);
    return ret;
  } else {
    return std::vector<mkldnn::memory::format>();
  }
}

struct TestArrayShapes {
  std::vector<nnvm::TShape> shapes;
  std::vector<mkldnn::memory::primitive_desc> pds;
};

static TestArrayShapes GetTestArrayShapes() {
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
    pds.push_back(GetMemPD(s1, dtype, formats[0]));
    pds.push_back(GetMemPD(s2, dtype, formats[1]));
  }
  {
    // 5D
    TShape s(5);
    s[0] = 96; s[1] = 1; s[2] = 3; s[3] = 11; s[4] = 11;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::goihw));

    std::vector<mkldnn::memory::format> formats = GetMKLDNNFormat(5, dtype);
    pds.push_back(GetMemPD(s, dtype, formats[0]));
  }

  TestArrayShapes ret;
  ret.shapes = shapes;
  ret.pds = pds;
  return ret;
}

TEST(MKLDNN_NDArray, GetDataReorder) {
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<TShape> shapes = tas.shapes;
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;


  // Reorder from the default to any other layout.
  for (auto s : shapes) {
    NDArray arr(s, Context());
    InitDefaultArray(&arr);
    for (auto pd : pds) {
      if (s.Size() == pd.get_size() / sizeof(mshadow::default_real_t)) {
        const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(pd);
        printf("reorder from (");
        for (size_t i = 0; i < s.ndim(); i++)
          printf("%ld, ", s[i]);
        printf(") to (");
        for (int i = 0; i < pd.desc().data.ndims; i++)
          printf("%d, ", pd.desc().data.dims[i]);
        printf("), format: %d\n", pd.desc().data.format);
        MKLDNNStream::Get()->Submit(false);
        VerifyMem(*mem);
        MKLDNNStream::Get()->Cleanup();
      }
    }
  }

  // Reorder from a special layout to another layout.
  for (auto s : shapes) {
    for (auto from_pd : pds) {
      if (from_pd.get_size() / sizeof(mshadow::default_real_t) == s.Size()) {
        NDArray arr(s, Context());
        // There is possibility that the dimensions of an NDArray doesn't match
        // with the MKLDNN memory inside.
        printf("Init array (");
        for (size_t i = 0; i < s.ndim(); i++)
          printf("%ld, ", s[i]);
        printf(") with MKLDNN memory (");
        for (int i = 0; i < from_pd.desc().data.ndims; i++)
          printf("%d, ", from_pd.desc().data.dims[i]);
        printf("), format: %d\n", from_pd.desc().data.format);
        InitMKLDNNArray(&arr, from_pd);
        for (auto to_pd : pds) {
          if (to_pd.get_size() / sizeof(mshadow::default_real_t) == s.Size()) {
            const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(to_pd);
            printf("reorder from (");
            for (size_t i = 0; i < s.ndim(); i++)
              printf("%ld, ", s[i]);
            printf("), format: %d to (",
                   arr.GetMKLDNNData()->get_primitive_desc().desc().data.format);
            for (int i = 0; i < to_pd.desc().data.ndims; i++)
              printf("%d, ", to_pd.desc().data.dims[i]);
            printf("), format: %d\n", to_pd.desc().data.format);
            MKLDNNStream::Get()->Submit(false);
            VerifyMem(*mem);
            MKLDNNStream::Get()->Cleanup();
          }
        }
      }
    }
  }
}

struct NDArrayAttrs {
  NDArray arr;
  std::string desc;
  NDArrayAttrs(NDArray arr, std::string desc) : arr(arr), desc(desc) {}
};

struct OpAttrs {
  nnvm::NodeAttrs attrs;
  std::vector<DispatchMode> dispatches;
  int num_inputs;
  int num_outputs;
};

OpAttrs GetCopyOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("_copy");
  attrs.num_inputs = 1;
  attrs.num_outputs = 1;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetCopyBackwardsOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("_backward_copy");
  attrs.num_inputs = 1;
  attrs.num_outputs = 1;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetReluOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("Activation");
  attrs.attrs.dict.insert({"act_type", "relu"});
  attrs.attrs.op->attr_parser(&attrs.attrs);
  attrs.num_inputs = 1;
  attrs.num_outputs = 1;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetReluBackwardsOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("_backward_Activation");
  attrs.attrs.dict.insert({"act_type", "relu"});
  attrs.attrs.op->attr_parser(&attrs.attrs);
  attrs.num_inputs = 2;
  attrs.num_outputs = 1;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetSumOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("elemwise_add");
  attrs.num_inputs = 2;
  attrs.num_outputs = 1;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetSumBackwardsOp() {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("_backward_add");
  attrs.num_inputs = 1;
  attrs.num_outputs = 2;
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetConcatOp(int num_args, int dim) {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("concat");
  attrs.num_inputs = num_args;
  attrs.num_outputs = 1;
  attrs.attrs.dict.insert({"num_args" , std::to_string(num_args)});
  attrs.attrs.dict.insert({"dim" , std::to_string(dim)});
  attrs.attrs.op->attr_parser(&attrs.attrs);
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetConcatBackwardsOp(int num_args, int dim) {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("_backward_Concat");
  attrs.num_inputs = 2;
  attrs.num_outputs = num_args;
  attrs.attrs.dict.insert({"num_args" , std::to_string(num_args)});
  attrs.attrs.dict.insert({"dim" , std::to_string(dim)});
  attrs.attrs.op->attr_parser(&attrs.attrs);
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}

OpAttrs GetPoolingOp(int kernel, int stride, int pad) {
  OpAttrs attrs;
  attrs.attrs.op = Op::Get("Pooling");
  attrs.num_inputs = 1;
  attrs.num_outputs = 1;
  attrs.attrs.dict.insert({"kernel" , std::to_string(kernel)});
  attrs.attrs.dict.insert({"stride" , std::to_string(stride)});
  attrs.attrs.dict.insert({"pad" , std::to_string(pad)});
  attrs.attrs.dict.insert({"pool_type" , "max"});
  attrs.attrs.op->attr_parser(&attrs.attrs);
  attrs.dispatches.resize(2);
  attrs.dispatches[0] = DispatchMode::kFCompute;
  attrs.dispatches[1] = DispatchMode::kFComputeEx;
  return attrs;
}


void PrintVerifyMsg(const NDArrayAttrs &arr1, const NDArrayAttrs &arr2) {
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
std::vector<NDArrayAttrs> GetTestInputArrays(bool rand = false, int num_inputs = 1, int dim = 0) {
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<nnvm::TShape> shapes = tas.shapes;
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  std::vector<NDArrayAttrs> in_arrs;
  std::string desc;

  int slice_amount = 1;
  if (dim == 0)
    slice_amount = num_inputs;
  for (auto shape : shapes) {
    if (dim >= shape.ndim())
      continue;
    shape[dim] = shape[dim] * num_inputs;

    // Type 1.
    NDArray arr(shape, Context());
    in_arrs.emplace_back(arr, "Normal NDArray");
    InitDefaultArray(&in_arrs.back().arr, rand);
    for (auto pd : pds) {
      if (num_inputs > 1) {
        // preserve if matching layout else just expand on 0 dim
        if (shape.ndim() == pd.desc().data.ndims)
          pd = GetExpandedMemPD(pd, num_inputs, dim);
        else
          pd = GetExpandedMemPD(pd, num_inputs);
      }

      if (shape.Size() != pd.get_size() / sizeof(mshadow::default_real_t))
        continue;

      // Type 2, 3.
      arr = NDArray(shape, Context());
      desc = "MKLDNN NDArray";
      if (shape.ndim() != pd.desc().data.ndims) {
        std::stringstream ss;
        ss << "MKLDNN NDArray with different memory layout " <<
           shape.ndim() << "/" << pd.desc().data.ndims;
        desc = ss.str();
      }
      InitMKLDNNArray(&arr, pd);
      in_arrs.emplace_back(arr, desc);

      // Type 4, 5, 6.
      arr = NDArray(shape, Context());
      desc = "Reshaped MKLDNN NDArray";
      if (shape.ndim() != pd.desc().data.ndims) {
        std::stringstream ss;
        ss << "Reshaped MKLDNN NDArray with different memory layout "
           << shape.ndim() << "/" << pd.desc().data.ndims;
        desc = ss.str();
      }
      InitMKLDNNArray(&arr, pd);
      in_arrs.emplace_back(arr.Slice(slice_amount, arr.shape()[0] - slice_amount), desc);
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
std::vector<NDArrayAttrs> GetTestOutputArrays(
    const TShape &shp,
    const std::vector<mkldnn::memory::primitive_desc> &pds,
    std::vector<float>scale = {1}) {
  TShape shape = shp;

  for (int dim = 0; dim < scale.size(); dim++)
    shape[dim] = static_cast<int>(shape[dim] * scale[dim]);

  std::vector<NDArrayAttrs> in_arrs;
  std::string desc;
  // Type 1.
  NDArray arr(shape, Context());
  in_arrs.emplace_back(arr, "Normal NDArray");
  InitDefaultArray(&in_arrs.back().arr, true);

  // Type 4.
  TShape tmp_shape = shape;
  tmp_shape[0] = shape[0] * 2;
  NDArray arr0(tmp_shape, Context());
  InitDefaultArray(&arr0, true);
  in_arrs.emplace_back(arr0.Slice(1, shape[0] + 1), "Reshaped NDArray");

  // Type 5.
  // Get a reused version.
  nnvm::TShape s(1);
  s[0] = shape.Size();
  NDArray arr1(s, Context());
  arr1 = arr1.AsArray(shape, arr1.dtype());
  InitDefaultArray(&arr1, true);
  in_arrs.emplace_back(arr1, "Reused NDArray");

  // Type 6.
  s[0] = shape.Size() * GetTypeSize(mshadow::default_type_flag);
  NDArray arr2(s, Context(), true, mshadow::kUint8);
  arr2 = arr2.AsArray(shape, mshadow::default_type_flag);
  InitDefaultArray(&arr2, true);
  in_arrs.emplace_back(arr2, "Reused NDArray with diff data type");

  // Type 7
  s[0] = shape.Size() * GetTypeSize(mshadow::default_type_flag) * 2;
  NDArray arr3(s, Context(), true, mshadow::kUint8);
  tmp_shape[0] = shape[0] * 2;
  arr3 = arr3.AsArray(tmp_shape, mshadow::default_type_flag);
  InitDefaultArray(&arr3, true);
  in_arrs.emplace_back(arr3.Slice(1, shape[0] + 1), "Reused+Reshaped NDArray");

  for (auto pd : pds) {
    if (shape.Size() != pd.get_size() / sizeof(mshadow::default_real_t))
      continue;

    for (int dim = 0; dim < scale.size(); dim++)
      pd = GetExpandedMemPD(pd, scale[dim], dim);


    // Type 2, 3.

    arr = NDArray(shape, Context());
    desc = "MKLDNN NDArray";
    if (shape.ndim() != pd.desc().data.ndims) {
      std::stringstream ss;
      ss << "MKLDNN NDArray with different memory layout "
         << shape.ndim() << "/" << pd.desc().data.ndims;
      desc = ss.str();
    }
    in_arrs.emplace_back(arr, desc);
    InitMKLDNNArray(&in_arrs.back().arr, pd, true);

    // Type 8, 9.
    // Get a reused version.
    nnvm::TShape s(1);
    s[0] = shape.Size();
    NDArray arr = NDArray(s, Context());
    arr = arr.AsArray(shape, arr.dtype());
    InitMKLDNNArray(&arr, pd, true);
    desc = "Reused MKLDNN NDArray";
    if (shape.ndim() != pd.desc().data.ndims) {
      std::stringstream ss;
      ss << "Reused MKLDNN NDArray with different memory layout "
         << shape.ndim() << "/" << pd.desc().data.ndims;
      desc = ss.str();
    }
    in_arrs.emplace_back(arr, desc);
  }
  return in_arrs;
}

TEST(MKLDNN_NDArray, GetTestInputArraysConcat) {
  auto in_arrs = GetTestInputArrays();
  for (int dim = 0; dim < 5; dim++) {
    for (int num_inputs = 2; num_inputs < 5; num_inputs++) {
      std::vector<NDArrayAttrs> expanded_arrs = GetTestInputArrays(false, num_inputs, dim);
      int i = 0;
      for (auto &arr : in_arrs) {
        if (dim >= arr.arr.shape().ndim())
          continue;
        auto ex_arr = expanded_arrs[i];
        PrintVerifyMsg(arr, ex_arr);
        EXPECT_EQ(arr.arr.shape().Size() * num_inputs, ex_arr.arr.shape().Size());
        EXPECT_EQ(arr.arr.shape()[dim] * num_inputs, ex_arr.arr.shape()[dim]);
        i++;
      }
    }
  }
}

TEST(MKLDNN_NDArray, GetTestOutputArraysConcat) {
  auto shapes_pds = GetTestArrayShapes();
  std::vector<nnvm::TShape> shapes; shapes = shapes_pds.shapes;
  std::vector<mkldnn::memory::primitive_desc> pds = shapes_pds.pds;
  for (auto &shape : shapes) {
    for (int dim = 0; dim < 5; dim++) {
      for (int num_inputs = 2; num_inputs < 5; num_inputs++) {
        if (shape.ndim() <= dim)
          continue;
        std::cout << "Extending " << shape << " dim " <<
                  dim << " and " << num_inputs << "num_inputs\n";
        std::vector<float> scale_vector(in_arr.arr.shape().ndim());
        for (int i = 0; i < in_arr.arr.shape().ndim(); i++)
          scale_vector[i] = 1;
        scale_vector[dim] = num_inputs;
        auto output_arrs = GetTestOutputArrays(shape, pds, scale_vector);
        for (auto &out_arr : output_arrs) {
          auto out_shape = out_arr.arr.shape();
          EXPECT_EQ(shape.Size() * num_inputs, out_arr.arr.shape().Size());
          EXPECT_EQ(shape[dim] * num_inputs, out_arr.arr.shape()[dim]);
        }
      }
    }
  }
}

void VerifyCopyResult(const std::vector<NDArray *> &in_arrs,
                      const std::vector<NDArray *> &out_arrs) {
  NDArray tmp1 = in_arrs[0]->Reorder2Default();
  NDArray tmp2 = out_arrs[0]->Reorder2Default();
  EXPECT_EQ(tmp1.shape().Size(), tmp2.shape().Size());
  TBlob d1 = tmp1.data();
  TBlob d2 = tmp2.data();
  EXPECT_EQ(memcmp(d1.dptr_, d2.dptr_,
                   tmp1.shape().Size() * sizeof(mshadow::default_real_t)), 0);
}

void VerifyActResult(const std::vector<NDArray *> &in_arrs,
                     const std::vector<NDArray *> &out_arrs) {
  NDArray tmp1 = in_arrs[0]->Reorder2Default();
  NDArray tmp2 = out_arrs[0]->Reorder2Default();
  TBlob blob1 = tmp1.data();
  TBlob blob2 = tmp2.data();
  mshadow::default_real_t *d1 = static_cast<mshadow::default_real_t*>(blob1.dptr_);
  mshadow::default_real_t *d2 = static_cast<mshadow::default_real_t*>(blob2.dptr_);
  EXPECT_EQ(tmp1.shape().Size(), tmp2.shape().Size());
  for (size_t i = 0; i < tmp1.shape().Size(); i++) {
    EXPECT_EQ(std::fmax(d1[i], 0), d2[i]);
  }
}

void VerifySumResult(const std::vector<NDArray *> &in_arrs,
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

void VerifyActBackwardsResult(const std::vector<NDArray *> &in_arrs,
                              const std::vector<NDArray *> &out_arrs) {
  NDArray tmp1 = in_arrs[0]->Reorder2Default();  // out grads
  NDArray tmp2 = in_arrs[1]->Reorder2Default();  // input
  NDArray tmp3 = out_arrs[0]->Reorder2Default();  // input grads
  TBlob blob1 = tmp1.data();
  TBlob blob2 = tmp2.data();
  TBlob blob3 = tmp3.data();
  mshadow::default_real_t *d1 = static_cast<mshadow::default_real_t*>(blob1.dptr_);
  mshadow::default_real_t *d2 = static_cast<mshadow::default_real_t*>(blob2.dptr_);
  mshadow::default_real_t *d3 = static_cast<mshadow::default_real_t*>(blob3.dptr_);
  EXPECT_EQ(tmp1.shape().Size(), tmp2.shape().Size());
  for (size_t i = 0; i < tmp1.shape().Size(); i++) {
    ASSERT_EQ(d2[i] > 0 ? d1[i] : 0, d3[i]);
  }
}

void VerifySumBackwardsResult(const std::vector<NDArray *> &in_arrs,
                               const std::vector<NDArray *> &out_arrs) {
  NDArray out_grads = in_arrs[0]->Reorder2Default();  // out grads
  NDArray input_grads1 = out_arrs[0]->Reorder2Default();  // input grads
  NDArray input_grads2 = out_arrs[1]->Reorder2Default();  // input grads
  mshadow::default_real_t *og = out_grads.data().dptr<mshadow::default_real_t>();
  mshadow::default_real_t *ig1 = input_grads1.data().dptr<mshadow::default_real_t>();
  mshadow::default_real_t *ig2 = input_grads2.data().dptr<mshadow::default_real_t>();
  for (size_t i = 0; i < out_grads.shape().Size(); i++) {
    ASSERT_EQ(og[i], ig1[i]);
    ASSERT_EQ(og[i], ig2[i]);
  }
}

/*
 * Determines axis ndarrays are concatenated by
 * Used to verify concat/concat backwards operator
 */
int GetDim(TShape input_shape, TShape output_shape) {
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
int GetBlockSize(TShape shape, int dim) {
  int block_size = 1;
  for (int i = shape.ndim() - 1; i >= dim; i--)
    block_size *= shape[i];
  return block_size;
}

void VerifyConcatResult(const std::vector<NDArray *> &in_arrs,
                        const std::vector<NDArray *> &out_arrs) {
  int num_inputs = in_arrs.size();
  int input_size = in_arrs[0]->shape().Size();
  TShape input_shape = in_arrs[0]->shape();
  NDArray output = out_arrs[0]->Reorder2Default();
  size_t total_size = output.shape().Size();
  EXPECT_EQ(input_size * num_inputs, total_size);
  mshadow::default_real_t *out_data = output.data().dptr<mshadow::default_real_t>();

  int dim = GetDim(input_shape, output.shape());
  int block_size = GetBlockSize(input_shape, dim);
  int num_blocks = input_size / block_size;
  for (size_t input_num = 0; input_num < num_inputs; input_num++) {
    NDArray tmp = in_arrs[input_num]->Reorder2Default();
    mshadow::default_real_t* data = tmp.data().dptr<mshadow::default_real_t>();
    for (size_t block_num = 0; block_num < num_blocks; block_num++) {
      for (size_t i = 0; i < block_size; i++)
        ASSERT_EQ(data[block_num * block_size + i],
                  out_data[(block_num * num_inputs + input_num) * block_size + i]);
    }
  }
}

void VerifyConcatBackwardsResult(const std::vector<NDArray *> &in_arrs,
                        const std::vector<NDArray *> &out_arrs) {
  // in_arrs is larger array, out_arr is ammler
  int num_inputs = out_arrs.size();
  int input_size = out_arrs[0]->shape().Size();
  TShape input_shape = out_arrs[0]->shape();
  NDArray output = in_arrs[0]->Reorder2Default();
  size_t total_size = output.shape().Size();
  EXPECT_EQ(input_size * num_inputs, total_size);
  mshadow::default_real_t *out_data = output.data().dptr<mshadow::default_real_t>();

  int dim = GetDim(input_shape, output.shape());
  int block_size = GetBlockSize(input_shape, dim);
  int num_blocks = input_size / block_size;
  for (size_t input_num = 0; input_num < num_inputs; input_num++) {
    NDArray tmp = out_arrs[input_num]->Reorder2Default();
    mshadow::default_real_t* data = tmp.data().dptr<mshadow::default_real_t>();
    for (size_t block_num = 0; block_num < num_blocks; block_num++) {
      for (size_t i = 0; i < block_size; i++)
        ASSERT_EQ(data[block_num * block_size + i],
                  out_data[(block_num * num_inputs + input_num) * block_size + i]);
    }
  }
}

TShape GetShiftedCoordinate(const TShape coordindate, int dim, int amount) {
  CHECK(dim < coordindate.ndim());
  TShape tmp = coordindate;
  CHECK(tmp[dim] + amount >= 0);
  tmp[dim] = tmp[dim] + amount;
  return tmp;
}

float GetValueAtCoordinate(const NDArray &in_arr, const TShape coordinate) {
  TShape input_shape = in_arr.shape();
  std::vector<int> block_sizes(input_shape.ndim()); // number of indexes must move to move along the axis
  for (int dim = 0; dim < input_shape.ndim(); dim++)
    block_sizes[dim] = GetBlockSize(input_shape, dim + 1);

  int index = 0;
  for (int i = 0; i < coordinate.ndim(); i++)
    index += block_sizes[i] * coordinate[i];
  return static_cast<float*>(in_arr.Reorder2Default().data().dptr_)[index];
}

// TODO: accept callback to handle operation
// center is formated as (N,C,everything else)
float PoolAtCoordinate(const NDArray &in_arr, const TShape coordinate, const TShape kernel_shape) {
  TShape input_shape = in_arr.shape();
  float max = -std::numeric_limits<float>::max();
  CHECK(input_shape[0] > coordinate[0]) << "Batch dimension should be within in_arr bounds";
  CHECK(input_shape[1] > coordinate[1]) << "Pooling dimension should be within in_arr bounds";

  // assumes the kernel is the last two dim
  for (int dim = 0; dim < kernel_shape.ndim(); dim++) {
    int center = coordinate[dim + 2];
    int shift = kernel_shape[dim] / 2;
    for (int i = -shift; i < kernel_shape[dim] - shift; i++) {
      float value = -std::numeric_limits<float>::max();
      if (center + i < 0 || center + i >= input_shape[dim + 2]) {
        value = -std::numeric_limits<float>::max(); // depends
      } else {
        TShape shifted_shape = GetShiftedCoordinate(coordinate, 2 + dim, i);
        value = GetValueAtCoordinate(in_arr, shifted_shape);
      }
      max = std::fmax(value, max);
    }
  }
  return max;
}

TEST(MKLDNN_NDArray, GetValueAtCoordinate) {
  TShape test_shape = {1,1,8};
  TShape kernel_shape = {3};
  NDArray arr(test_shape, Context());
  InitDefaultArray(&arr);
  TShape coord1 = {0,0,0}; // edge
  TShape coord2 = {0,0,7}; // edge
  TShape coord3 = {0,0,4}; //middle
  for (int i = 0; i < 8; i++) {
    TShape tmp = {0,0,i};
    EXPECT_EQ(i - 4, GetValueAtCoordinate(arr, tmp));
  }
}

TEST(MKLDNN_NDArray, PoolAtCoordinate) {

  // one channel
  {
    TShape test_shape = {1,1,8};
    TShape odd_kernel_shape = {3};
    TShape even_kernel_shape = {4};
    NDArray arr(test_shape, Context());
    InitDefaultArray(&arr);
    TShape coord1 = {0,0,0}; // edge
    TShape coord2 = {0,0,7}; // edge
    TShape coord3 = {0,0,4}; // middle
    EXPECT_EQ(-3, PoolAtCoordinate(arr, coord1, odd_kernel_shape));
    EXPECT_EQ(3, PoolAtCoordinate(arr, coord2, odd_kernel_shape));
    EXPECT_EQ(1, PoolAtCoordinate(arr, coord3, odd_kernel_shape));
    EXPECT_EQ(-3, PoolAtCoordinate(arr, coord1, even_kernel_shape));
    EXPECT_EQ(3, PoolAtCoordinate(arr, coord2, even_kernel_shape));
    EXPECT_EQ(1, PoolAtCoordinate(arr, coord3, even_kernel_shape));
  }

  // two channels
  {
    TShape test_shape = {1,2,8};
    TShape odd_kernel_shape = {3};
    NDArray arr(test_shape, Context());
    InitDefaultArray(&arr);
    TShape coord1 = {0,0,0}; // edge
    TShape coord2 = {0,0,7}; // edge
    TShape coord3 = {0,0,4}; // middle
    TShape coord4 = {0,1,0}; // edge
    TShape coord5 = {0,1,7}; // edge
    TShape coord6 = {0,1,4}; // middle
    EXPECT_EQ(-7, PoolAtCoordinate(arr, coord1, odd_kernel_shape));
    EXPECT_EQ(-1, PoolAtCoordinate(arr, coord2, odd_kernel_shape));
    EXPECT_EQ(-3, PoolAtCoordinate(arr, coord3, odd_kernel_shape));
    EXPECT_EQ(1, PoolAtCoordinate(arr, coord4, odd_kernel_shape));
    EXPECT_EQ(7, PoolAtCoordinate(arr, coord5, odd_kernel_shape));
    EXPECT_EQ(5, PoolAtCoordinate(arr, coord6, odd_kernel_shape));
  }
}

TEST(MKLDNN_NDArray, GetShiftedCoordinate) {
  TShape test_shape = {1,1,1,1,1};
  TShape shifted_shape = GetShiftedCoordinate(test_shape, 2, 5);
  EXPECT_EQ(1, shifted_shape[0]);
  EXPECT_EQ(1, shifted_shape[1]);
  EXPECT_EQ(6, shifted_shape[2]);
  EXPECT_EQ(1, shifted_shape[3]);
  EXPECT_EQ(1, shifted_shape[4]);
}

void VerifyPool1D(const NDArray &input,
                  const NDArray &output,
                  const TShape padding,
                  const TShape kernel,
                  const TShape stride,
                  int batch_num,
                  int channel_num) {
  TShape input_shape = input.shape();
  TShape ptr(input_shape.ndim());
  ptr[0] = batch_num;
  ptr[1] = channel_num;
  ptr[2] = 0;
  mshadow::default_real_t* out_data = output.data().dptr<mshadow::default_real_t>();
  int out_ptr = 0;
  int lower = -padding[0];
  int upper = input_shape[3] + padding[0];
  for (int i = lower; i < upper - kernel[0]; i = i + stride[0]) {
    int center = i + kernel[0] / 2;
    TShape coordinate = GetShiftedCoordinate(ptr, 2, center);
    ASSERT_EQ(PoolAtCoordinate(input, coordinate, kernel), out_data[out_ptr]);
    out_ptr++;
  }
}

void VerifyPool2D(const NDArray &input,
                  const NDArray &output,
                  const TShape padding,
                  const TShape kernel,
                  const TShape stride,
                  int batch_num,
                  int channel_num) {
  TShape input_shape = input.shape();
  TShape ptr(input_shape.ndim());
  ptr[0] = batch_num;
  ptr[1] = channel_num;
  ptr[2] = 0;
  ptr[3] = 0;
  mshadow::default_real_t* out_data = output.data().dptr<mshadow::default_real_t>();
  int out_ptr = 0;
  int lower = -padding[0];
  int upper = input_shape[3] + padding[0];
  for (int i = lower; i < upper - kernel[0]; i = i + stride[0]) {
    int center = i + kernel[0] / 2;
    TShape coordinate = GetShiftedCoordinate(ptr, 3, center);
    for (int j = lower; j < upper - kernel[1]; j = j + stride[1]) {
      int center = j + kernel[1] / 2;
      TShape coordinate = GetShiftedCoordinate(coordinate, 4, center);
      ASSERT_EQ(PoolAtCoordinate(input, coordinate, kernel), out_data[out_ptr]);
      out_ptr++;
    }
  }
}

void VerifyPool3D(const NDArray &input,
                  const NDArray &output,
                  const TShape padding,
                  const TShape kernel,
                  const TShape stride,
                  int batch_num,
                  int channel_num) {
  TShape input_shape = input.shape();
  TShape ptr(input_shape.ndim());
  ptr[0] = batch_num;
  ptr[1] = channel_num;
  ptr[2] = 0;
  ptr[3] = 0;
  ptr[4] = 0;
  mshadow::default_real_t* out_data = output.data().dptr<mshadow::default_real_t>();
  int out_ptr = 0;
  int lower = -padding[0];
  int upper = input_shape[3] + padding[0];
  for (int i = lower; i < upper - kernel[0]; i = i + stride[0]) {
    int center = i + kernel[0] / 2;
    TShape coordinate = GetShiftedCoordinate(ptr, 3, center);
    for (int j = lower; j < upper - kernel[1]; j = j + stride[1]) {
      int center = j + kernel[1] / 2;
      TShape coordinate = GetShiftedCoordinate(coordinate, 4, center);
      for (int k = lower; k < upper - kernel[2]; k = k + stride[2]) {
        int center = j + kernel[2] / 2;
        TShape coordinate = GetShiftedCoordinate(coordinate, 5, center);
        ASSERT_EQ(PoolAtCoordinate(input, coordinate, kernel), out_data[out_ptr]);
        out_ptr++;
      }
    }
  }
}

void VerifyPoolingResult(const std::vector<NDArray *> &in_arrs,
                         const std::vector<NDArray *> &out_arrs,
                         const OpAttrs &attrs) {
  mxnet::op::PoolingParam param;
  param.Init(attrs.attrs.dict);
  TShape kernel = param.kernel;
  TShape padding = param.pad;
  TShape stride = param.stride;
  int pool_type; // max
  NDArray input = in_arrs[0]->Reorder2Default();
  NDArray output = out_arrs[0]->Reorder2Default();
//  mshadow::default_real_t* out_data = output.data().dptr<mshadow::default_real_t>();
  TShape input_shape = input.shape();
  CHECK(input_shape.ndim() > 2);
  int num_batches = input_shape[0];
  int num_channels = input_shape[1];


  for (int batch_num = 0; batch_num < num_batches; batch_num++) {
    for (int channel_num = 0; channel_num < num_channels; channel_num++) {
      if (input_shape.ndim() == 3)
        VerifyPool1D(input, output, padding, kernel, stride, batch_num, channel_num);
      if (input_shape.ndim() == 4)
        VerifyPool2D(input, output, padding, kernel, stride, batch_num, channel_num);
      if (input_shape.ndim() == 5)
        VerifyPool3D(input, output, padding, kernel, stride, batch_num, channel_num);
    }
  }
}

TEST(MKLDNN_NDArray, VerifyPoolingResult) {
  TShape test_shape = {1,1,3};

  std::vector<NDArray *> in_arrs(1);
  std::vector<NDArray *> out_arrs(1);
  NDArray arr(test_shape, Context());
  InitDefaultArray(&arr);
  mshadow::default_real_t *input_data = arr.data().dptr<mshadow::default_real_t>();
  EXPECT_EQ(-1, input_data[0]);
  EXPECT_EQ(0, input_data[1]);
  EXPECT_EQ(1, input_data[2]);

  // test base
  {
    OpAttrs attrs;
    attrs.attrs.op = Op::Get("Pooling");
    attrs.attrs.dict.insert({"kernel", "1"});
    attrs.attrs.dict.insert({"stride", "1"});
    attrs.attrs.dict.insert({"pad", "0"});
    attrs.attrs.dict.insert({"pool_type", "max"});
    attrs.attrs.op->attr_parser(&attrs.attrs);
    NDArray expected_output(test_shape, Context());
    mshadow::default_real_t *expected_data = expected_output.data().dptr<mshadow::default_real_t>();
    expected_data[0] = -1;
    expected_data[1] = 0;
    expected_data[2] = 1;
    in_arrs[0] = &arr;
    out_arrs[0] = &expected_output;
    VerifyPoolingResult(in_arrs, out_arrs, attrs);
  }

  // test kernel
  {
    OpAttrs attrs;
    attrs.attrs.op = Op::Get("Pooling");
    attrs.attrs.dict.insert({"kernel" , "2"});
    attrs.attrs.dict.insert({"stride" , "1"});
    attrs.attrs.dict.insert({"pad" , "0" });
    attrs.attrs.dict.insert({"pool_type" , "max"});
    attrs.attrs.op->attr_parser(&attrs.attrs);
    TShape expected_shape = {1,1,2};
    NDArray expected_output(expected_shape, Context());
    mshadow::default_real_t* expected_data = expected_output.data().dptr<mshadow::default_real_t>();
    expected_data[0] = 0;
    expected_data[1] = 1;
    in_arrs[0] = &arr;
    out_arrs[0] = &expected_output;
    VerifyPoolingResult(in_arrs, out_arrs, attrs);
  }

  //test padding
  {
    OpAttrs attrs;
    attrs.attrs.op = Op::Get("Pooling");
    attrs.attrs.dict.insert({"kernel" , "2"});
    attrs.attrs.dict.insert({"stride" , "1"});
    attrs.attrs.dict.insert({"pad" , "1" });
    attrs.attrs.dict.insert({"pool_type" , "max"});
    attrs.attrs.op->attr_parser(&attrs.attrs);
    NDArray expected_output(test_shape, Context());
    mshadow::default_real_t* expected_data = expected_output.data().dptr<mshadow::default_real_t>();
    expected_data[0] = -1;
    expected_data[1] = 0;
    expected_data[2] = 1;
    in_arrs[0] = &arr;
    out_arrs[0] = &expected_output;
    VerifyPoolingResult(in_arrs, out_arrs, attrs);
  }

  //test stride
  {
    OpAttrs attrs;
    attrs.attrs.op = Op::Get("Pooling");
    attrs.attrs.dict.insert({"kernel" , "1"});
    attrs.attrs.dict.insert({"stride" , "2"});
    attrs.attrs.dict.insert({"pad" , "0" });
    attrs.attrs.dict.insert({"pool_type" , "max"});
    attrs.attrs.op->attr_parser(&attrs.attrs);
    TShape expected_shape = {1,1,1};
    NDArray expected_output(expected_shape, Context());
    mshadow::default_real_t* expected_data = expected_output.data().dptr<mshadow::default_real_t>();
    expected_data[0] = 0;
    in_arrs[0] = &arr;
    out_arrs[0] = &expected_output;
    VerifyPoolingResult(in_arrs, out_arrs, attrs);
  }
}

void VerifyAddRequest(const std::vector<NDArray*> &in_arrs,
                      const std::vector<NDArray*> &original_outputs,
                      const std::vector<NDArray*> &new_outputs,
                      VerifyFunc verify_fn) {
  NDArray tmp = new_outputs[0]->Reorder2Default() - original_outputs[0]->Reorder2Default();
  tmp.WaitToRead();
  verify_fn(in_arrs, {&tmp});
}

TEST(MKLDNN_NDArray, CopyFrom) {
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  for (auto &in_arr : in_arrs) {
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView())
      continue;
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), pds);
    for (auto &out_arr : out_arrs) {
      const mkldnn::memory *mem = in_arr.arr.GetMKLDNNData();
      out_arr.arr.CopyFrom(*mem);
      MKLDNNStream::Get()->Submit();
      std::vector<NDArray *> inputs(1);
      inputs[0] = &in_arr.arr;
      VerifyCopyResult(inputs, {&out_arr.arr});
    }
  }
}

void TestOp(const OpAttrs &attrs, VerifyFunc verify_fn) {
  std::vector<NDArray*> inputs(attrs.num_inputs);
  std::vector<NDArray*> outputs(attrs.num_outputs);
  std::vector<OpReqType> req(attrs.num_outputs);
  std::vector<DispatchMode> dispatches = attrs.dispatches;

  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  for (auto &in_arr : in_arrs) {
    for (auto &dispatch : dispatches) {
      std::vector<std::vector<NDArrayAttrs>> out_arrs(attrs.num_outputs);
      for (int i = 0; i < attrs.num_outputs; i++)
        out_arrs[i] = GetTestOutputArrays(in_arr.arr.shape(), pds);
      for (int i = 0; i < attrs.num_inputs; i++)
        inputs[i] = &in_arr.arr;
      for (size_t output_i = 0; output_i < out_arrs[0].size(); output_i++) {
        for (int i = 0; i < attrs.num_outputs; i++) {
          req[i] = kWriteTo;
          outputs[i] = &out_arrs[i][output_i].arr;
        }
        PrintVerifyMsg(in_arr, out_arrs[0][output_i]);
        Imperative::Get()->InvokeOp(Context(), attrs.attrs, inputs,
                                    outputs, req, dispatch, mxnet::OpStatePtr());
        Engine::Get()->WaitForAll();
        verify_fn(inputs, outputs);
      }
    }
  }

  for (auto &dispatch : dispatches) {
    in_arrs = GetTestInputArrays();
    for (auto &arr : in_arrs) {
      // If the array is a view, we shouldn't write data to it.
      if (arr.arr.IsView())
        continue;
      NDArrayAttrs orig(arr.arr.Copy(arr.arr.ctx()), "InPlace Copy");
      for (int i = 0; i < attrs.num_inputs; i++)
        inputs[i] = &arr.arr;
      for (int i = 0; i < attrs.num_outputs; i++) {
        req[i] = kWriteInplace;
        outputs[i] = &arr.arr;
      }
      PrintVerifyMsg(orig, arr);
      Imperative::Get()->InvokeOp(Context(), attrs.attrs, inputs, outputs, req,
                                  dispatch, mxnet::OpStatePtr());
      Engine::Get()->WaitForAll();
      std::vector<NDArray *> orig_inputs(attrs.num_inputs);
      for (int i = 0; i < attrs.num_inputs; i++)
        orig_inputs[i] = &orig.arr;
      verify_fn(orig_inputs, outputs);
    }
  }
}

void TestConcatOp(const OpAttrs &attrs, VerifyFunc verify_fn,
            bool backwards = false) {
  std::vector<NDArray*> inputs(attrs.num_inputs);
  std::vector<NDArray*> outputs(attrs.num_outputs);
  std::vector<OpReqType> req(attrs.num_outputs);
  std::vector<DispatchMode> dispatches = attrs.dispatches;

  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();

  // concat backwards uses scaled up inputs
  if (backwards) {
    std::string str_dim = const_cast<OpAttrs&>(attrs).attrs.dict["dim"];
    int dim = std::stoi(str_dim);
    in_arrs = GetTestInputArrays(false, attrs.num_outputs, dim);
  }

  for (auto &in_arr : in_arrs) {
    for (auto &dispatch : dispatches) {
      std::vector<std::vector<NDArrayAttrs>> out_arrs(attrs.num_outputs);

      std::string str_dim = const_cast<OpAttrs&>(attrs).attrs.dict["dim"];
      int dim = std::stoi(str_dim);
      if (dim >= in_arr.arr.shape().ndim())
        continue;
      float scale = backwards ? 1 / static_cast<float>(attrs.num_outputs) :
          static_cast<float>(attrs.num_inputs);

      std::vector<float> scale_vector(in_arr.arr.shape().ndim());
      for (int i = 0; i < in_arr.arr.shape().ndim(); i++)
        scale_vector[i] = 1;
      scale_vector[dim] = scale;

      for (int i = 0; i < attrs.num_outputs; i++)
        out_arrs[i] = GetTestOutputArrays(in_arr.arr.shape(), pds, scale_vector);

      for (int i = 0; i < attrs.num_inputs; i++)
        inputs[i] = &in_arr.arr;

      for (size_t output_i = 0; output_i < out_arrs[0].size(); output_i++) {
        for (int i = 0; i < attrs.num_outputs; i++) {
          req[i] = kWriteTo;
          outputs[i] = &out_arrs[i][output_i].arr;
        }

        PrintVerifyMsg(in_arr, out_arrs[0][output_i]);
        Imperative::Get()->InvokeOp(Context(), attrs.attrs, inputs,
                                    outputs, req, dispatch, mxnet::OpStatePtr());
        Engine::Get()->WaitForAll();
        verify_fn(inputs, outputs);
      }
    }
  }
}

int CalculateWidth(int width, int kernel, int padding, int stride) {
  return (width - kernel + 2 * padding) / (stride + 1);
}

void TestPoolingOp(const OpAttrs &attrs,
                  bool backwards = false) {
  std::vector<NDArray*> inputs(attrs.num_inputs);
  std::vector<NDArray*> outputs(attrs.num_outputs);
  std::vector<OpReqType> req(attrs.num_outputs);
  std::vector<DispatchMode> dispatches = attrs.dispatches;

  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  mxnet::op::PoolingParam param;
  param.Init(attrs.attrs.dict);
  TShape kernel = param.kernel;
  TShape padding = param.pad;
  TShape stride = param.stride;

  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();

  // concat backwards uses scaled up inputs
  if (backwards) {
    std::string str_dim = const_cast<OpAttrs&>(attrs).attrs.dict["dim"];
    int dim = std::stoi(str_dim);
    in_arrs = GetTestInputArrays(false, attrs.num_outputs, dim);
  }

  for (auto &in_arr : in_arrs) {
    // can only pool only 3D and 4D inputs
    if (in_arr.arr.shape().ndim() <= 2)
      continue;
    TShape input_shape = in_arr.arr.shape();
    for (auto &dispatch : dispatches) {
      std::vector<std::vector<NDArrayAttrs>> out_arrs(attrs.num_outputs);
      std::vector<float> scale_vector(in_arr.arr.shape().ndim());
      for (int i = 0; i < in_arr.arr.shape().ndim(); i++) {
        if (i < 2)
          scale_vector[i] = 1;
        else
          scale_vector[i] = CalculateWidth(input_shape[i], kernel[i-2], padding[i-2], stride[i-2]) / input_shape[i];
      }

      for (int i = 0; i < attrs.num_outputs; i++)
        out_arrs[i] = GetTestOutputArrays(in_arr.arr.shape(), pds, scale_vector);

      for (int i = 0; i < attrs.num_inputs; i++)
        inputs[i] = &in_arr.arr;

      for (size_t output_i = 0; output_i < out_arrs[0].size(); output_i++) {
        for (int i = 0; i < attrs.num_outputs; i++) {
          req[i] = kWriteTo;
          outputs[i] = &out_arrs[i][output_i].arr;
        }

        PrintVerifyMsg(in_arr, out_arrs[0][output_i]);
        Imperative::Get()->InvokeOp(Context(), attrs.attrs, inputs,
                                    outputs, req, dispatch, mxnet::OpStatePtr());
        Engine::Get()->WaitForAll();
        VerifyPoolingResult(inputs, outputs, attrs);
      }
    }
  }
}

TEST(IMPERATIVE, CopyOp) {
  OpAttrs attrs = GetCopyOp();
  TestOp(attrs, VerifyCopyResult);
}

TEST(IMPERATIVE, CopyBackwardsOp) {
  OpAttrs attrs = GetCopyBackwardsOp();
  TestOp(attrs, VerifyCopyResult);
}

TEST(IMPERATIVE, ActOp) {
  OpAttrs attrs = GetReluOp();
  TestOp(attrs, VerifyActResult);
}

TEST(IMPERATIVE, ActBackwardsOp) {
  OpAttrs attrs = GetReluBackwardsOp();
  TestOp(attrs, VerifyActBackwardsResult);
}

TEST(IMPERATIVE, SumOp) {
  OpAttrs attrs = GetSumOp();
  TestOp(attrs, VerifySumResult);
}

TEST(IMPERATIVE, SumBackwardsOp) {
  OpAttrs attrs = GetSumBackwardsOp();
  TestOp(attrs, VerifySumBackwardsResult);
}

TEST(IMPERATIVE, ConcatOp) {
  for (int num_inputs = 2; num_inputs < 4; num_inputs++) {
    for (int dim = 0; dim < 5; dim++) {
      OpAttrs attrs = GetConcatOp(num_inputs, dim);
      TestConcatOp(attrs, VerifyConcatResult);
    }
  }
}

TEST(IMPERATIVE, ConcatBackwardsOp) {
  for (int num_inputs = 2; num_inputs < 4; num_inputs++) {
    for (int dim = 0; dim < 5; dim++) {
      OpAttrs attrs = GetConcatBackwardsOp(num_inputs, dim);
      TestConcatOp(attrs, VerifyConcatBackwardsResult, true);
    }
  }
}

std::vector<TShape> GetInputKernelShapes(int dim, int max_size) {
  std::vector<TShape> kernels;
  for (int i = 1; i <= max_size; i++) {
    TShape tmp(dim);
    for (int j = 0; j < tmp.ndim(); j++)
      tmp[j] = i;
    kernels.push_back(tmp);
  }
  return kernels;
}

TEST(IMPERATIVE, PoolingOp) {
  // TODO: change kernel, stride, pad
  for (int kernel = 1; kernel < 3; kernel++) {
    for (int stride = 1; stride < 2; stride++) {
      for (int pad = 0; pad < 1; pad++) {
        OpAttrs attrs = GetPoolingOp(kernel, stride, pad);
        TestPoolingOp(attrs, false);
      }
    }
  }
}

TEST(MKLDNN_BASE, MKLDNNSum) {
  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  std::vector<NDArrayAttrs> in_arrs2 = GetTestInputArrays(true);
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;

  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), pds);
    for (auto &out_arr : out_arrs) {
      auto in_mem1 = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      if (out_arr.arr.IsView())
        continue;
      auto out_mem = out_arr.arr.GetMKLDNNData();
      PrintVerifyMsg(in_arr, in_arr);
      op::MKLDNNSum(*in_mem1, *in_mem2, *out_mem);
      MKLDNNStream::Get()->Submit();
      VerifySumResult({&in_arr.arr, &in_arr2.arr}, {&out_arr.arr});
    }
  }

  // in place
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_primitive_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    op::MKLDNNSum(*input_mem, *input_mem2, *input_mem);
    MKLDNNStream::Get()->Submit();
    VerifySumResult({&orig_arr.arr, &in_arr2.arr}, {&in_arr.arr});
  }
}

TEST(MKLDNN_BASE, CreateMKLDNNMem) {
  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  std::vector<NDArrayAttrs> in_arrs2 = GetTestInputArrays(true);
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::primitive_desc> pds = tas.pds;
  MKLDNNStream *stream = MKLDNNStream::Get();

  // kWriteTo
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), pds);
    for (auto &out_arr : out_arrs) {
      auto in_mem = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      NDArray orig_output = out_arr.arr.Copy(out_arr.arr.ctx());
      orig_output.WaitToRead();
      PrintVerifyMsg(in_arr, out_arr);
      auto out_mem = out_arr.arr.GetMKLDNNData();
      auto output_mem_t = CreateMKLDNNMem(out_arr.arr, out_mem->get_primitive_desc(), kWriteTo);
      op::MKLDNNSum(*in_mem, *in_mem2, *output_mem_t.second);
      CommitOutput(out_arr.arr, output_mem_t);
      stream->Submit();
      VerifySumResult({&in_arr.arr, &in_arr2.arr}, {&out_arr.arr});
    }
  }

  // kWriteInPlace
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    orig_arr.arr.WaitToRead();
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_primitive_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    auto output_mem_t = CreateMKLDNNMem(in_arr.arr,
        input_mem->get_primitive_desc(), kWriteInplace, &in_arr.arr);
    op::MKLDNNSum(*input_mem, *input_mem2, *output_mem_t.second);
    CommitOutput(in_arr.arr, output_mem_t);
    stream->Submit();
    VerifySumResult({&orig_arr.arr, &in_arr2.arr}, {&in_arr.arr});
  }

  // kAddTo
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), pds);
    for (auto &out_arr : out_arrs) {
      auto in_mem = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      NDArray orig_output = out_arr.arr.Copy(out_arr.arr.ctx());
      orig_output.WaitToRead();
      PrintVerifyMsg(in_arr, out_arr);
      auto out_mem = out_arr.arr.GetMKLDNNData();
      auto output_mem_t = CreateMKLDNNMem(out_arr.arr, out_mem->get_primitive_desc(), kAddTo);
      op::MKLDNNSum(*in_mem, *in_mem2, *output_mem_t.second);
      CommitOutput(out_arr.arr, output_mem_t);
      stream->Submit();
      VerifyAddRequest(
          {&in_arr.arr, &in_arr2.arr}, {&orig_output}, {&out_arr.arr}, VerifySumResult);
    }
  }

  // kNullOp
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    orig_arr.arr.WaitToRead();
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_primitive_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    auto output_mem_t = CreateMKLDNNMem(in_arr.arr, input_mem->get_primitive_desc(), kNullOp);
    op::MKLDNNSum(*input_mem, *input_mem2, *output_mem_t.second);
    CommitOutput(in_arr.arr, output_mem_t);
    stream->Submit();
    // original and input should be the same since noop
    VerifyCopyResult({&orig_arr.arr}, {&in_arr.arr});
  }
}

#endif
