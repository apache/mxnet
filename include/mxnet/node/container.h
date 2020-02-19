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
 * \brief Array container
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_NODE_CONTAINER_H_
#define MXNET_NODE_CONTAINER_H_

#include <mxnet/node/node.h>

#include <type_traits>
#include <vector>
#include <initializer_list>
#include <unordered_map>
#include <utility>
#include <string>

namespace mxnet {

/*! \brief array node content in array */
class ArrayNode : public Object {
 public:
  /*! \brief the data content */
  std::vector<ObjectRef> data;

  static constexpr const char* _type_key = "Array";
  MXNET_DECLARE_FINAL_OBJECT_INFO(ArrayNode, Object);
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template<typename Converter,
         typename TIter>
class IterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;   // NOLINT(*)
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit IterAdapter(TIter iter) : iter_(iter) {}
  inline IterAdapter& operator++() {
    ++iter_;
    return *this;
  }
  inline IterAdapter operator+(difference_type offset) const {
    return IterAdapter(iter_ + offset);
  }

  template<typename T = IterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type
  inline operator-(const IterAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  inline bool operator==(IterAdapter other) const {
    return iter_ == other.iter_;
  }
  inline bool operator!=(IterAdapter other) const {
    return !(*this == other);
  }
  inline const value_type operator*() const {
    return Converter::convert(*iter_);
  }

 private:
  TIter iter_;
};

/*!
 * \brief Array container of NodeRef in DSL graph.
 *  Array implements copy on write semantics, which means array is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam T The content NodeRef type.
 */
template<typename T,
         typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type >
class Array : public ObjectRef {
 public:
  /*!
   * \brief default constructor
   */
  Array() {
    data_ = make_object<ArrayNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Array(Array<T> && other) {  // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Array(const Array<T> &other) { // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Array(runtime::ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  Array(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Array(std::initializer_list<T> init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Array(const std::vector<T>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit Array(size_t n, const T& val) {
    auto tmp_node = make_object<ArrayNode>();
    for (size_t i = 0; i < n; ++i) {
      tmp_node->data.push_back(val);
    }
    data_ = std::move(tmp_node);
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(Array<T> && other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(const Array<T> & other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_object<ArrayNode>();
    for (IterType it = begin; it != end; ++it) {
      n->data.push_back(T(*it));
    }
    data_ = std::move(n);
  }
  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  inline const T operator[](size_t i) const {
    return DowncastNoCheck<T>(
        static_cast<const ArrayNode*>(data_.get())->data[i]);
  }
  /*! \return The size of the array */
  inline size_t size() const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const ArrayNode*>(data_.get())->data.size();
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline ArrayNode* CopyOnWrite() {
    if (data_.get() == nullptr || !data_.unique())  {
      runtime::ObjectPtr<ArrayNode> n = make_object<ArrayNode>();
      n->data = static_cast<ArrayNode*>(data_.get())->data;
      runtime::ObjectPtr<Object>(std::move(n)).swap(data_);
    }
    return static_cast<ArrayNode*>(data_.get());
  }
  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  inline void push_back(const T& item) {
    ArrayNode* n = this->CopyOnWrite();
    n->data.push_back(item);
  }
  /*!
   * \brief Resize the array.
   * \param size The new size.
   */
  inline void resize(size_t size) {
    ArrayNode* n = this->CopyOnWrite();
    n->data.resize(size);
  }
  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  inline void Set(size_t i, const T& value) {
    ArrayNode* n = this->CopyOnWrite();
    n->data[i] = value;
  }
  /*! \return whether array is empty */
  inline bool empty() const {
    return size() == 0;
  }
  /*!
   * \brief Helper function to apply fmutate to mutate an array.
   * \param fmutate The transformation function T -> T.
   * \tparam F the type of the mutation function.
   * \note This function performs copy on write optimization.
   */
  template<typename F>
  inline void MutateByApply(F fmutate) {
    ArrayNode* ptr = static_cast<ArrayNode*>(data_.get());
    if (ptr == nullptr) return;
    if (data_.unique()) {
      // Copy on write optimization.
      // Perform inplace update because this is an unique copy.
      for (size_t i = 0; i < ptr->data.size(); ++i) {
        // It is important to use move here
        // to make prevent the element's ref count from increasing
        // so fmutate itself can perform copy-on-write optimization
        T old_elem = DowncastNoCheck<T>(std::move(ptr->data[i]));
        T new_elem = fmutate(std::move(old_elem));
        ptr->data[i] = std::move(new_elem);
      }
    } else {
      // lazily trigger copy if there is element change.
      runtime::ObjectPtr<ArrayNode> copy;
      for (size_t i = 0; i < ptr->data.size(); ++i) {
        T old_elem = DowncastNoCheck<T>(ptr->data[i]);
        T new_elem = fmutate(old_elem);
        if (!new_elem.same_as(ptr->data[i])) {
          // copy the old array
          if (copy == nullptr) {
            copy = runtime::make_object<ArrayNode>(*ptr);
          }
          copy->data[i] = std::move(new_elem);
        }
      }
      // replace the data with the new copy.
      if (copy != nullptr) {
        data_ = std::move(copy);
      }
    }
  }

  /*! \brief specify container node */
  using ContainerType = ArrayNode;

  struct ValueConverter {
    using ResultType = T;
    static inline T convert(const ObjectRef& n) {
      return DowncastNoCheck<T>(n);
    }
  };
  using iterator = IterAdapter<ValueConverter,
                               std::vector<ObjectRef>::const_iterator>;

  using reverse_iterator = IterAdapter<
    ValueConverter,
    std::vector<ObjectRef>::const_reverse_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const ArrayNode*>(data_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const ArrayNode*>(data_.get())->data.end());
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return reverse_iterator(static_cast<const ArrayNode*>(data_.get())->data.rbegin());
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return reverse_iterator(static_cast<const ArrayNode*>(data_.get())->data.rend());
  }
};

}  // namespace mxnet
#endif  // MXNET_NODE_CONTAINER_H_
