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
 * \file ffi_helper
 * \brief Helper class to support additional objects in FFI.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_FFI_HELPER_H_
#define MXNET_RUNTIME_FFI_HELPER_H_

#include <mxnet/runtime/object.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/memory.h>
#include <limits>

namespace mxnet {
namespace runtime {

/*! \brief Ellipsis. */
class EllipsisObj : public Object {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kEllipsis;
  static constexpr const char* _type_key = "MXNet.Ellipsis";
  MXNET_DECLARE_FINAL_OBJECT_INFO(EllipsisObj, Object);
};

inline ObjectRef CreateEllipsis() {
  return ObjectRef(make_object<EllipsisObj>());
}

/*! \brief Slice. */
class SliceObj : public Object {
 public:
  int64_t start;
  int64_t stop;
  int64_t step;

  static constexpr const uint32_t _type_index = TypeIndex::kSlice;
  static constexpr const char* _type_key = "MXNet.Slice";
  MXNET_DECLARE_FINAL_OBJECT_INFO(SliceObj, Object);
};

class Slice : public ObjectRef {
 public:
  explicit inline Slice(int64_t start, int64_t stop, int64_t step,
                        ObjectPtr<SliceObj>&& data = make_object<SliceObj>()) {
    data->start = start;
    data->stop = stop;
    data->step = step;
    data_ = std::move(data);
  }

  explicit inline Slice(int64_t stop)
      : Slice(kNoneValue, stop, kNoneValue) {
  }

  // constant to represent None.
  static constexpr int64_t kNoneValue = std::numeric_limits<int64_t>::min();

  MXNET_DEFINE_OBJECT_REF_METHODS(Slice, ObjectRef, SliceObj);
};

int64_t inline SliceNoneValue() {
  return Slice::kNoneValue;
}

class IntegerObj: public Object {
 public:
  int64_t value;
  static constexpr const uint32_t _type_index = TypeIndex::kInteger;
  static constexpr const char* _type_key = "MXNet.Integer";
  MXNET_DECLARE_FINAL_OBJECT_INFO(IntegerObj, Object);
};

class Integer: public ObjectRef {
 public:
  explicit Integer(int64_t value,
                   ObjectPtr<IntegerObj>&& data = make_object<IntegerObj>()) {
    data->value = value;
    data_ = std::move(data);
  }
  MXNET_DEFINE_OBJECT_REF_METHODS(Integer, ObjectRef, IntegerObj);
};

//  Helper functions for fast FFI implementations
/*!
 * \brief A builder class that helps to incrementally build ADT.
 */
class ADTBuilder {
 public:
  /*! \brief default constructor */
  ADTBuilder() = default;

  explicit inline ADTBuilder(uint32_t tag, uint32_t size)
      : data_(make_inplace_array_object<ADTObj, ObjectRef>(size)) {
    data_->size = size;
  }

  template <typename... Args>
  void inline EmplaceInit(size_t idx, Args&&... args) {
    data_->EmplaceInit(idx, std::forward<Args>(args)...);
  }

  ADT inline Get() {
    return ADT(std::move(data_));
  }

 private:
  friend class ADT;
  ObjectPtr<ADTObj> data_;
};
}  // namespace runtime
}  // namespace mxnet
#endif  // MXNET_RUNTIME_FFI_HELPER_H_
