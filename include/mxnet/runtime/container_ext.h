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
 * \file container_ext.h
 * \brief Common POD(plain old data) container types extension.
 */
// Acknowledgement: This file originates from dgl
#ifndef MXNET_RUNTIME_CONTAINER_EXT_H_
#define MXNET_RUNTIME_CONTAINER_EXT_H_
#include <dmlc/logging.h>
#include <mxnet/runtime/memory.h>
#include <mxnet/runtime/object.h>

#include <string_view>
#include <string>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>

namespace mxnet {
namespace runtime {

// Forward declare MXNetArgValue
class MXNetArgValue;

/*! \brief String-aware ObjectRef hash functor */
struct ObjectRefHash {
  /*!
   * \brief Calculate the hash code of an ObjectRef
   * \param a The given ObjectRef
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
  size_t operator()(const ObjectRef& a) const;
};

/*! \brief String-aware ObjectRef equal functor */
struct ObjectRefEqual {
  /*!
   * \brief Check if the two ObjectRef are equal
   * \param a One ObjectRef
   * \param b The other ObjectRef
   * \return String equality if both are strings, pointer address equality otherwise.
   */
  bool operator()(const ObjectRef& a, const ObjectRef& b) const;
};

/*! \brief Shared content of all specializations of hash map */
class MapObj : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  /*! \brief Type of the values in the hash map */
  using mapped_type = ObjectRef;
  /*! \brief Type of the actual underlying container */
  using ContainerType = std::unordered_map<ObjectRef, ObjectRef, ObjectRefHash, ObjectRefEqual>;
  /*! \brief Iterator class */
  using iterator = ContainerType::iterator;
  /*! \brief Iterator class */
  using const_iterator = ContainerType::const_iterator;
  /*! \brief Type of value stored in the hash map */
  using KVType = ContainerType::value_type;

  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kMXNetMap;
  static constexpr const char* _type_key = "MXNet.Map";
  MXNET_DECLARE_FINAL_OBJECT_INFO(MapObj, Object);

  /*!
   * \brief Number of elements in the MapObj
   * \return The result
   */
  size_t size() const { return data_.size(); }
  /*!
   * \brief Count the number of times a key exists in the hash map
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const { return data_.count(key); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const { return data_.at(key); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) { return data_.at(key); }
  /*! \return begin iterator */
  iterator begin() { return data_.begin(); }
  /*! \return const begin iterator */
  const_iterator begin() const { return data_.begin(); }
  /*! \return end iterator */
  iterator end() { return data_.end(); }
  /*! \return end iterator */
  const_iterator end() const { return data_.end(); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  const_iterator find(const key_type& key) const { return data_.find(key); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) { return data_.find(key); }
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) { data_.erase(position); }
  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { data_.erase(key); }
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<MapObj> Empty() { return make_object<MapObj>(); }

 protected:
  /*!
   * \brief Create the map using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
  template <typename IterType>
  static ObjectPtr<Object> CreateFromRange(IterType first, IterType last) {
    ObjectPtr<MapObj> p = make_object<MapObj>();
    p->data_ = ContainerType(first, last);
    return p;
  }
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map) {
    MapObj* map_node = static_cast<MapObj*>(map->get());
    map_node->data_[kv.first] = kv.second;
  }
  /*!
   * \brief Create an empty container with elements copying from another MapObj
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<MapObj> CopyFrom(MapObj* from) {
    ObjectPtr<MapObj> p = make_object<MapObj>();
    p->data_ = ContainerType(from->data_.begin(), from->data_.end());
    return p;
  }
  /*! \brief The real container storing data */
  ContainerType data_;
  template <typename, typename, typename, typename>
  friend class Map;
};

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
class Map : public ObjectRef {
 public:
  using key_type = K;
  using mapped_type = V;
  class iterator;
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapObj::Empty(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) { data_ = std::move(other.data_); }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {}
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    data_ = MapObj::CreateFromRange(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapObj::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapObj::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const { return DowncastNoCheck<V>(GetMapObj()->at(key)); }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : GetMapObj()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapObj::InsertMaybeReHash(MapObj::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapObj()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapObj()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapObj()->find(key)); }

  void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  MapObj* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapObj::Empty();
    } else if (!data_.unique()) {
      data_ = MapObj::CopyFrom(GetMapObj());
    }
    return GetMapObj();
  }
  /*! \brief specify container node */
  using ContainerType = MapObj;

  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(DowncastNoCheck<K>(kv.first), DowncastNoCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

   private:
    iterator(const MapObj::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename, typename>
    friend class Map;

    MapObj::iterator itr;
  };

 private:
  /*! \brief Return data_ as type of pointer of MapObj */
  MapObj* GetMapObj() const { return static_cast<MapObj*>(data_.get()); }
};

/*!
 * \brief Merge two Maps.
 * \param lhs the first Map to merge.
 * \param rhs the second Map to merge.
 * @return The merged Array. Original Maps are kept unchanged.
 */
template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
inline Map<K, V> Merge(Map<K, V> lhs, const Map<K, V>& rhs) {
  for (const auto& p : rhs) {
    lhs.Set(p.first, p.second);
  }
  return std::move(lhs);
}

/*! \brief An object representing string. It's POD type. */
class StringObj : public Object {
 public:
  /*! \brief The pointer to string data. */
  const char* data;

  /*! \brief The length of the string object. */
  uint64_t size;

  static constexpr const uint32_t _type_index = TypeIndex::kMXNetString;
  static constexpr const char* _type_key = "MXNet.String";
  MXNET_DECLARE_FINAL_OBJECT_INFO(StringObj, Object);

 private:
  /*! \brief String object which is moved from std::string container. */
  class FromStd;

  friend class String;
};

/*!
 * \brief Reference to string objects.
 *
 * \code
 *
 * // Example to create runtime String reference object from std::string
 * std::string s = "hello world";
 *
 * // You can create the reference from existing std::string
 * String ref{std::move(s)};
 *
 * // You can rebind the reference to another string.
 * ref = std::string{"hello world2"};
 *
 * // You can use the reference as hash map key
 * std::unordered_map<String, int32_t> m;
 * m[ref] = 1;
 *
 * // You can compare the reference object with other string objects
 * assert(ref == "hello world", true);
 *
 * // You can convert the reference to std::string again
 * string s2 = (string)ref;
 *
 * \endcode
 */
class String : public ObjectRef {
 public:
  /*!
   * \brief Construct an empty string.
   */
  String() : String(std::string()) {}
  /*!
   * \brief Construct a new String object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  String(std::string other);  // NOLINT(*)

  /*!
   * \brief Construct a new String object
   *
   * \param other a char array.
   */
  String(const char* other)  // NOLINT(*)
      : String(std::string(other)) {}

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   *
   */
  inline String& operator=(std::string other);

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   */
  inline String& operator=(const char* other);

  /*!
   * \brief Compares this String object to other
   *
   * \param other The String to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The string to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const std::string& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this to other
   *
   * \param other The character array to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const char* other) const {
    return memncmp(data(), other, size(), std::stold(other));
  }

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const char* c_str() const { return get()->data; }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const {
    const auto* ptr = get();
    return ptr->size;
  }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t length() const { return size(); }

  /*!
   * \brief Retun if the string is empty
   *
   * \return true if empty, false otherwise.
   */
  bool empty() const { return size() == 0; }

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return get()->data; }

  /*!
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{get()->data, size()}; }

  /*!
   * \brief Check if a MXNetArgValue can be converted to String, i.e. it can be std::string or String
   * \param val The value to be checked
   * \return A boolean indicating if val can be converted to String
   */
  inline static bool CanConvertFrom(const MXNetArgValue& val);

  /*!
   * \brief Hash the binary bytes
   * \param data The data pointer
   * \param size The size of the bytes.
   * \return the hash value.
   */
  static size_t HashBytes(const char* data, size_t size) {
    // This function falls back to string copy with c++11 compiler and is
    // recommended to be compiled with c++14
    return std::hash<std::string_view>()(std::string_view(data, size));
  }

  MXNET_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(String, ObjectRef, StringObj);

 private:
  /*!
   * \brief Compare two char sequence
   *
   * \param lhs Pointers to the char array to compare
   * \param rhs Pointers to the char array to compare
   * \param lhs_count Length of the char array to compare
   * \param rhs_count Length of the char array to compare
   * \return int zero if both char sequences compare equal. negative if this
   * appear before other, positive otherwise.
   */
  static int memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count);

  /*!
   * \brief Concatenate two char sequences
   *
   * \param lhs Pointers to the lhs char array
   * \param lhs_size The size of the lhs char array
   * \param rhs Pointers to the rhs char array
   * \param rhs_size The size of the rhs char array
   *
   * \return The concatenated char sequence
   */
  static String Concat(const char* lhs, size_t lhs_size, const char* rhs, size_t rhs_size) {
    std::string ret(lhs, lhs_size);
    ret.append(rhs, rhs_size);
    return String(ret);
  }

  // Overload + operator
  friend String operator+(const String& lhs, const String& rhs);
  friend String operator+(const String& lhs, const std::string& rhs);
  friend String operator+(const std::string& lhs, const String& rhs);
  friend String operator+(const String& lhs, const char* rhs);
  friend String operator+(const char* lhs, const String& rhs);

  friend struct mxnet::runtime::ObjectRefEqual;
};

/*! \brief An object representing string moved from std::string. */
class StringObj::FromStd : public StringObj {
 public:
  /*!
   * \brief Construct a new FromStd object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  explicit FromStd(std::string other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::string data_container;

  friend class String;
};

inline String::String(std::string other) {
  auto ptr = make_object<StringObj::FromStd>(std::move(other));
  ptr->size = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

inline String& String::operator=(std::string other) {
  String replace{std::move(other)};
  data_.swap(replace.data_);
  return *this;
}

inline String& String::operator=(const char* other) { return operator=(std::string(other)); }

inline String operator+(const String& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const std::string& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const std::string& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const char* lhs, const String& rhs) {
  size_t lhs_size = std::stold(lhs);
  size_t rhs_size = rhs.size();
  return String::Concat(lhs, lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const char* rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = std::stold(rhs);
  return String::Concat(lhs.data(), lhs_size, rhs, rhs_size);
}

// Overload < operator
inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

inline bool operator<(const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const String& lhs, const char* rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const char* lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

// Overload > operator
inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

inline bool operator>(const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const String& lhs, const char* rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const char* lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

// Overload <= operator
inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

inline bool operator<=(const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const String& lhs, const char* rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const char* lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

// Overload >= operator
inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

inline bool operator>=(const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const String& lhs, const char* rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(rhs) <= 0; }

// Overload == operator
inline bool operator==(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const String& lhs, const char* rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const char* lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

// Overload != operator
inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const String& lhs, const char* rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const char* lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
}

inline int String::memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
  if (lhs == rhs && lhs_count == rhs_count) return 0;

  for (size_t i = 0; i < lhs_count && i < rhs_count; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs_count < rhs_count) {
    return -1;
  } else if (lhs_count > rhs_count) {
    return 1;
  } else {
    return 0;
  }
}

inline size_t ObjectRefHash::operator()(const ObjectRef& a) const {
  if (const auto* str = a.as<StringObj>()) {
    return String::HashBytes(str->data, str->size);
  }
  return ObjectHash()(a);
}

inline bool ObjectRefEqual::operator()(const ObjectRef& a, const ObjectRef& b) const {
  if (a.same_as(b)) {
    return true;
  }
  if (const auto* str_a = a.as<StringObj>()) {
    if (const auto* str_b = b.as<StringObj>()) {
      return String::memncmp(str_a->data, str_b->data, str_a->size, str_b->size) == 0;
    }
  }
  return false;
}

}  // namespace runtime
}  // namespace mxnet

#endif  // MXNET_RUNTIME_CONTAINER_EXT_H_
