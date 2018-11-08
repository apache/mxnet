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
 * Copyright (c) 2015 by Contributors
 * \file serialization.h
 * \brief Serialization of some STL and nnvm data-structures
 * \author Clement Fuji Tsang
 */

#ifndef MXNET_COMMON_SERIALIZATION_H_
#define MXNET_COMMON_SERIALIZATION_H_

#include <dmlc/logging.h>
#include <mxnet/graph_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


namespace mxnet {
namespace common {

template<typename T>
inline size_t SerializedSize(const T &obj);

template<typename T>
inline size_t SerializedSize(const nnvm::Tuple <T> &obj);

template<typename K, typename V>
inline size_t SerializedSize(const std::map <K, V> &obj);

template<>
inline size_t SerializedSize(const std::string &obj);

template<typename... Args>
inline size_t SerializedSize(const std::tuple<Args...> &obj);

template<typename T>
inline void Serialize(const T &obj, char **buffer);

template<typename T>
inline void Serialize(const nnvm::Tuple <T> &obj, char **buffer);

template<typename K, typename V>
inline void Serialize(const std::map <K, V> &obj, char **buffer);

template<>
inline void Serialize(const std::string &obj, char **buffer);

template<typename... Args>
inline void Serialize(const std::tuple<Args...> &obj, char **buffer);

template<typename T>
inline void Deserialize(T *obj, const std::string &buffer, size_t *curr_pos);

template<typename T>
inline void Deserialize(nnvm::Tuple <T> *obj, const std::string &buffer, size_t *curr_pos);

template<typename K, typename V>
inline void Deserialize(std::map <K, V> *obj, const std::string &buffer, size_t *curr_pos);

template<>
inline void Deserialize(std::string *obj, const std::string &buffer, size_t *curr_pos);

template<typename... Args>
inline void Deserialize(std::tuple<Args...> *obj, const std::string &buffer, size_t *curr_pos);


template<typename T>
struct is_container {
  static const bool value = !std::is_pod<T>::value;
};

template<typename T>
inline size_t SerializedSize(const T &obj) {
  return sizeof(T);
}

template<typename T>
inline size_t SerializedSize(const nnvm::Tuple <T> &obj) {
  if (is_container<T>::value) {
    size_t sum_val = 4;
    for (const auto& el : obj) {
      sum_val += SerializedSize(el);
    }
    return sum_val;
  } else {
    return 4 + (obj.ndim() * sizeof(T));
  }
}

template<typename K, typename V>
inline size_t SerializedSize(const std::map <K, V> &obj) {
  size_t sum_val = 4;
  if (is_container<K>::value && is_container<V>::value) {
    for (const auto& p : obj) {
      sum_val += SerializedSize(p.first) + SerializedSize(p.second);
    }
  } else if (is_container<K>::value) {
    for (const auto& p : obj) {
      sum_val += SerializedSize(p.first);
    }
    sum_val += sizeof(V) * obj.size();
  } else if (is_container<V>::value) {
    for (const auto& p : obj) {
      sum_val += SerializedSize(p.second);
    }
    sum_val += sizeof(K) * obj.size();
  } else {
    sum_val += (sizeof(K) + sizeof(V)) * obj.size();
  }
  return sum_val;
}

template<>
inline size_t SerializedSize(const std::string &obj) {
  return obj.size() + 4;
}

template<int I>
struct serialized_size_tuple {
  template<typename... Args>
  static inline size_t Compute(const std::tuple<Args...> &obj) {
    return SerializedSize(std::get<I>(obj)) + serialized_size_tuple<I-1>::Compute(obj);
  }
};

template<>
struct serialized_size_tuple<0> {
  template<typename... Args>
  static inline size_t Compute(const std::tuple<Args...> &obj) {
    return SerializedSize(std::get<0>(obj));
  }
};

template<typename... Args>
inline size_t SerializedSize(const std::tuple<Args...> &obj) {
  return serialized_size_tuple<sizeof... (Args)-1>::Compute(obj);
}

//  Serializer

template<typename T>
inline size_t SerializedContainerSize(const T &obj, char **buffer) {
  uint32_t size = obj.size();
  std::memcpy(*buffer, &size, 4);
  *buffer += 4;
  return (size_t) size;
}

template<typename T>
inline void Serialize(const T &obj, char **buffer) {
  std::memcpy(*buffer, &obj, sizeof(T));
  *buffer += sizeof(T);
}

template<typename T>
inline void Serialize(const nnvm::Tuple <T> &obj, char **buffer) {
  uint32_t size = obj.ndim();
  std::memcpy(*buffer, &size, 4);
  *buffer += 4;
  for (auto& el : obj) {
    Serialize(el, buffer);
  }
}

template<typename K, typename V>
inline void Serialize(const std::map <K, V> &obj, char **buffer) {
  SerializedContainerSize(obj, buffer);
  for (auto& p : obj) {
    Serialize(p.first, buffer);
    Serialize(p.second, buffer);
  }
}

template<>
inline void Serialize(const std::string &obj, char **buffer) {
  auto size = SerializedContainerSize(obj, buffer);
  std::memcpy(*buffer, &obj[0], size);
  *buffer += size;
}

template<int I>
struct serialize_tuple {
  template<typename... Args>
  static inline void Compute(const std::tuple<Args...> &obj, char **buffer) {
    serialize_tuple<I-1>::Compute(obj, buffer);
    Serialize(std::get<I>(obj), buffer);
  }
};

template<>
struct serialize_tuple<0> {
  template<typename... Args>
  static inline void Compute(const std::tuple<Args...> &obj, char **buffer) {
    Serialize(std::get<0>(obj), buffer);
  }
};

template<typename... Args>
inline void Serialize(const std::tuple<Args...> &obj, char **buffer) {
  serialize_tuple<sizeof... (Args)-1>::Compute(obj, buffer);
}

// Deserializer

template<typename T>
inline size_t DeserializedContainerSize(T *obj, const std::string &buffer, size_t *curr_pos) {
  uint32_t size = obj->size();
  std::memcpy(&size, &buffer[*curr_pos], 4);
  *curr_pos += 4;
  return (size_t) size;
}

template<typename T>
inline void Deserialize(T *obj, const std::string &buffer, size_t *curr_pos) {
  std::memcpy(obj, &buffer[*curr_pos], sizeof(T));
  *curr_pos += sizeof(T);
}

template<typename T>
inline void Deserialize(nnvm::Tuple <T> *obj, const std::string &buffer, size_t *curr_pos) {
  uint32_t size = obj->ndim();
  std::memcpy(&size, &buffer[*curr_pos], 4);
  *curr_pos += 4;
  obj->SetDim(size);
  for (size_t i = 0; i < size; ++i) {
    Deserialize((*obj)[i], buffer, curr_pos);
  }
}

template<typename K, typename V>
inline void Deserialize(std::map <K, V> *obj, const std::string &buffer, size_t *curr_pos) {
  auto size = DeserializedContainerSize(obj, buffer, curr_pos);
  K first;
  for (size_t i = 0; i < size; ++i) {
    Deserialize(&first, buffer, curr_pos);
    Deserialize(&(*obj)[first], buffer, curr_pos);
  }
}

template<>
inline void Deserialize(std::string *obj, const std::string &buffer, size_t *curr_pos) {
  auto size = DeserializedContainerSize(obj, buffer, curr_pos);
  obj->resize(size);
  std::memcpy(&(obj->front()), &buffer[*curr_pos], size);
  *curr_pos += size;
}

template<int I>
struct deserialize_tuple {
  template<typename... Args>
  static inline void Compute(std::tuple<Args...> *obj,
                             const std::string &buffer, size_t *curr_pos) {
    deserialize_tuple<I-1>::Compute(obj, buffer, curr_pos);
    Deserialize(&std::get<I>(*obj), buffer, curr_pos);
  }
};

template<>
struct deserialize_tuple<0> {
  template<typename... Args>
  static inline void Compute(std::tuple<Args...> *obj,
                             const std::string &buffer, size_t *curr_pos) {
    Deserialize(&std::get<0>(*obj), buffer, curr_pos);
  }
};

template<typename... Args>
inline void Deserialize(std::tuple<Args...> *obj, const std::string &buffer, size_t *curr_pos) {
  deserialize_tuple<sizeof... (Args)-1>::Compute(obj, buffer, curr_pos);
}


template<typename T>
inline void Serialize(const T& obj, std::string* serialized_data) {
  serialized_data->resize(SerializedSize(obj));
  char* curr_pos = &(serialized_data->front());
  Serialize(obj, &curr_pos);
  CHECK_EQ((int64_t)curr_pos - (int64_t)&(serialized_data->front()),
           serialized_data->size());
}

template<typename T>
inline void Deserialize(T* obj, const std::string& serialized_data) {
  size_t curr_pos = 0;
  Deserialize(obj, serialized_data, &curr_pos);
  CHECK_EQ(curr_pos, serialized_data.size());
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_SERIALIZATION_H_
