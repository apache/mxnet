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

#include "gtest/gtest.h"
#include "../../src/operator/nn/mkldnn/mkldnn_base-inl.h"

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
  CHECK_EQ(mkldnn_format_last, 56);
  CHECK_EQ(mkldnn_nchw, 5);
  CHECK_EQ(mkldnn_oihw, 12);
}

static void InitArray(NDArray &arr) {
  const TBlob &blob = arr.data();
  mshadow::default_real_t *data = blob.dptr<mshadow::default_real_t>();
  size_t size = blob.Size();
  for (size_t i = 0; i < size; i++)
    data[i] = i;
}

static void VerifyMem(const mkldnn::memory &mem) {
  mkldnn::memory::primitive_desc pd = mem.get_primitive_desc();
  if (pd.desc().data.format == GetDefaultFormat(pd.desc())) {
    mshadow::default_real_t *data
        = static_cast<mshadow::default_real_t *>(mem.get_data_handle());
    size_t size = pd.get_size() / sizeof(mshadow::default_real_t);
    for (size_t i = 0; i < size; i++)
      CHECK_EQ(data[i], static_cast<mshadow::default_real_t>(i));
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

TEST(MKLDNN_GET_DATA_REORDER, DataReorder) {
  std::vector<TShape> shapes;
  std::vector<mkldnn::memory::primitive_desc> pds;
  int dtype = mshadow::DataType<mshadow::default_real_t>::kFlag;
  {
    // 1D
    TShape s(1);
    s[0] = 100;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::x));
  }
  {
    // 2D
    TShape s(2);
    s[0] = 25;
    s[1] = 4;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::nc));
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::io));
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::oi));
  }
  {
    // 4D
    TShape s(4);
    s[0] = 5;
    s[1] = 5;
    s[2] = 2;
    s[3] = 2;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::nchw));
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::oihw));
  }
  {
    // 5D
    TShape s(5);
    s[0] = 5;
    s[1] = 5;
    s[2] = 2;
    s[3] = 2;
    s[4] = 1;
    shapes.push_back(s);
    pds.push_back(GetMemPD(s, dtype, mkldnn::memory::format::goihw));
  }

  // Reorder from the default to any other layout.
  for (auto s : shapes) {
    NDArray arr(s, Context());
    InitArray(arr);
    for (auto pd : pds) {
      const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(pd);
      VerifyMem(*mem);
    }
  }

  // Reorder from a special layout to another layout.
//  for (auto s : shapes) {
//    NDArray arr(s, Context());
//    InitMKLDNNArray(arr);
//    for (auto pd : pds) {
//      const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(pd);
//      VerifyMem(*mem);
//    }
//  }
}

#endif
