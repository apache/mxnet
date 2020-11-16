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
/*
 * \file data_type.h
 * \brief Primitive runtime data type.
 */
// Acknowledgement: This file originates from incubator-tvm
// Acknowledgement: MXNetDataType structure design originates from Halide.
#ifndef MXNET_RUNTIME_DATA_TYPE_H_
#define MXNET_RUNTIME_DATA_TYPE_H_

#include <mxnet/runtime/c_runtime_api.h>
#include <dmlc/logging.h>
#include <type_traits>


namespace mxnet {
namespace runtime {
/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of MXNetDataType in compiler to store quick hint
 */
class MXNetDataType {
 public:
  /*! \brief Type code for the MXNetDataType. */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = MXNetTypeCode::kHandle,
  };
  /*! \brief default constructor */
  MXNetDataType() {}
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit MXNetDataType(DLDataType dtype)
      : data_(dtype) {}
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   */
  MXNetDataType(int code, int bits, int lanes) {
    data_.code = static_cast<uint8_t>(code);
    data_.bits = static_cast<uint8_t>(bits);
    data_.lanes = static_cast<uint16_t>(lanes);
  }
  /*! \return The type code. */
  int code() const {
    return static_cast<int>(data_.code);
  }
  /*! \return number of bits in the data. */
  int bits() const {
    return static_cast<int>(data_.bits);
  }
  /*! \return number of bytes to store each scalar. */
  int bytes() const {
    return (bits() + 7) / 8;
  }
  /*! \return number of lanes in the data. */
  int lanes() const {
    return static_cast<int>(data_.lanes);
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const {
    return lanes() == 1;
  }
  /*! \return whether type is a scalar type. */
  bool is_bool() const {
    return code() == MXNetDataType::kUInt && bits() == 1;
  }
  /*! \return whether type is a float type. */
  bool is_float() const {
    return code() == MXNetDataType::kFloat;
  }
  /*! \return whether type is an int type. */
  bool is_int() const {
    return code() == MXNetDataType::kInt;
  }
  /*! \return whether type is an uint type. */
  bool is_uint() const {
    return code() == MXNetDataType::kUInt;
  }
  /*! \return whether type is a handle type. */
  bool is_handle() const {
    return code() == MXNetDataType::kHandle;
  }
  /*! \return whether type is a vector type. */
  bool is_vector() const {
    return lanes() > 1;
  }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  MXNetDataType with_lanes(int lanes) const {
    return MXNetDataType(data_.code, data_.bits, lanes);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  MXNetDataType with_bits(int bits) const {
    return MXNetDataType(data_.code, bits, data_.lanes);
  }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  MXNetDataType element_of() const {
    return with_lanes(1);
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator==(const MXNetDataType& other) const {
    return
        data_.code == other.data_.code &&
        data_.bits == other.data_.bits &&
        data_.lanes == other.data_.lanes;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator!=(const MXNetDataType& other) const {
    return !operator==(other);
  }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType () const {
    return data_;
  }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static MXNetDataType Int(int bits, int lanes = 1) {
    return MXNetDataType(kDLInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static MXNetDataType UInt(int bits, int lanes = 1) {
    return MXNetDataType(kDLUInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static MXNetDataType Float(int bits, int lanes = 1) {
    return MXNetDataType(kDLFloat, bits, lanes);
  }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static MXNetDataType Bool(int lanes = 1) {
    return MXNetDataType::UInt(1, lanes);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static MXNetDataType Handle(int bits = 64, int lanes = 1) {
    return MXNetDataType(kHandle, bits, lanes);
  }

 private:
  DLDataType data_;
};

}  // namespace runtime

using MXNetDataType = runtime::MXNetDataType;

}  // namespace mxnet
#endif  //  MXNET_RUNTIME_DATA_TYPE_H_
