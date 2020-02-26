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
 * \file container.h
 * \brief Common POD(plain old data) container types.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_CONTAINER_H_
#define MXNET_RUNTIME_CONTAINER_H_
#include <dmlc/logging.h>
#include <mxnet/runtime/memory.h>
#include <mxnet/runtime/object.h>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace mxnet {
namespace runtime {

class ADTBuilder;
/*!
 * \brief Base template for classes with array like memory layout.
 *
 *        It provides general methods to access the memory. The memory
 *        layout is ArrayType + [ElemType]. The alignment of ArrayType
 *        and ElemType is handled by the memory allocator.
 *
 * \tparam ArrayType The array header type, contains object specific metadata.
 * \tparam ElemType The type of objects stored in the array right after
 * ArrayType.
 *
 * \code
 * // Example usage of the template to define a simple array wrapper
 * class ArrayObj : public InplaceArrayBase<ArrayObj, Elem> {
 * public:
 *  // Wrap EmplaceInit to initialize the elements
 *  template <typename Iterator>
 *  void Init(Iterator begin, Iterator end) {
 *   size_t num_elems = std::distance(begin, end);
 *   auto it = begin;
 *   this->size = 0;
 *   for (size_t i = 0; i < num_elems; ++i) {
 *     InplaceArrayBase::EmplaceInit(i, *it++);
 *     this->size++;
 *   }
 *  }
 * }
 *
 * void test_function() {
 *   vector<Elem> fields;
 *   auto ptr = make_inplace_array_object<ArrayObj, Elem>(fields.size());
 *   ptr->Init(fields.begin(), fields.end());
 *
 *   // Access the 0th element in the array.
 *   assert(ptr->operator[](0) == fields[0]);
 * }
 *
 * \endcode
 */
template <typename ArrayType, typename ElemType>
class InplaceArrayBase {
 public:
  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Const reference to ElemType at the index.
   */
  const ElemType& operator[](size_t idx) const {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
  ElemType& operator[](size_t idx) {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Destroy the Inplace Array Base object
   */
  ~InplaceArrayBase() {
    if (!(std::is_standard_layout<ElemType>::value &&
          std::is_trivial<ElemType>::value)) {
      size_t size = Self()->GetSize();
      for (size_t i = 0; i < size; ++i) {
        ElemType* fp = reinterpret_cast<ElemType*>(AddressOf(i));
        fp->ElemType::~ElemType();
      }
    }
  }

 protected:
  friend class ADTBuilder;
  /*!
   * \brief Construct a value in place with the arguments.
   *
   * \tparam Args Type parameters of the arguments.
   * \param idx Index of the element.
   * \param args Arguments to construct the new value.
   *
   * \note Please make sure ArrayType::GetSize returns 0 before first call of
   * EmplaceInit, and increment GetSize by 1 each time EmplaceInit succeeds.
   */
  template <typename... Args>
  void EmplaceInit(size_t idx, Args&&... args) {
    void* field_ptr = AddressOf(idx);
    new (field_ptr) ElemType(std::forward<Args>(args)...);
  }

 private:
  /*!
   * \brief Return the self object for the array.
   *
   * \return Pointer to ArrayType.
   */
  inline ArrayType* Self() const {
    return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
  }

  /*!
   * \brief Return the raw pointer to the element at idx.
   *
   * \param idx The index of the element.
   * \return Raw pointer to the element.
   */
  void* AddressOf(size_t idx) const {
    static_assert(alignof(ArrayType) % alignof(ElemType) == 0 &&
                      sizeof(ArrayType) % alignof(ElemType) == 0,
                  "The size and alignment of ArrayType should respect "
                  "ElemType's alignment.");

    size_t kDataStart = sizeof(ArrayType);
    ArrayType* self = Self();
    char* data_start = reinterpret_cast<char*>(self) + kDataStart;
    return data_start + idx * sizeof(ElemType);
  }
};

/*! \brief An object representing a structure or enumeration. */
class ADTObj : public Object, public InplaceArrayBase<ADTObj, ObjectRef> {
 public:
  /*! \brief The tag representing the constructor used. */
  uint32_t tag;
  /*! \brief Number of fields in the ADT object. */
  uint32_t size{0};
  // The fields of the structure follows directly in memory.

  static constexpr const uint32_t _type_index = TypeIndex::kMXNetADT;
  static constexpr const char* _type_key = "MXNet.ADT";
  MXNET_DECLARE_FINAL_OBJECT_INFO(ADTObj, Object);

 private:
  /*!
   * \return The number of elements in the array.
   */
  size_t GetSize() const { return size; }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \tparam Iterator Iterator type of the array.
   * \param begin The begin iterator.
   * \param end The end iterator.
   */
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = 0;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      // Only increment size after the initialization succeeds
      this->size++;
    }
  }

  friend class ADT;
  friend InplaceArrayBase<ADTObj, ObjectRef>;
};

/*! \brief reference to algebraic data type objects. */
class ADT : public ObjectRef {
 public:
  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param fields The fields of the ADT object.
   * \return The constructed ADT object reference.
   */
  ADT(uint32_t tag, std::vector<ObjectRef> fields)
      : ADT(tag, fields.begin(), fields.end()){};

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param begin The begin iterator to the start of the fields array.
   * \param end The end iterator to the end of the fields array.
   * \return The constructed ADT object reference.
   */
  template <typename Iterator>
  ADT(uint32_t tag, Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    auto ptr = make_inplace_array_object<ADTObj, ObjectRef>(num_elems);
    ptr->tag = tag;
    ptr->Init(begin, end);
    data_ = std::move(ptr);
  }

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param init The initializer list of fields.
   * \return The constructed ADT object reference.
   */
  ADT(uint32_t tag, std::initializer_list<ObjectRef> init)
      : ADT(tag, init.begin(), init.end()){};

  /*!
   * \brief Access element at index.
   *
   * \param idx The array index
   * \return const ObjectRef
   */
  const ObjectRef& operator[](size_t idx) const {
    return operator->()->operator[](idx);
  }

  /*!
   * \brief Return the ADT tag.
   */
  size_t tag() const { return operator->()->tag; }

  /*!
   * \brief Return the number of fields.
   */
  size_t size() const { return operator->()->size; }

  /*!
   * \brief Construct a tuple object.
   *
   * \tparam Args Type params of tuple feilds.
   * \param args Tuple fields.
   * \return ADT The tuple object reference.
   */
  template <typename... Args>
  static ADT Tuple(Args&&... args) {
    return ADT(0, std::forward<Args>(args)...);
  }

  MXNET_DEFINE_OBJECT_REF_METHODS(ADT, ObjectRef, ADTObj);
};

}  // namespace runtime
}  // namespace mxnet

#endif  // MXNET_RUNTIME_CONTAINER_H_
