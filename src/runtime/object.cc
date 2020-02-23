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
 * \file object.cc
 * \brief Object type management system.
 */
// Acknowledgement: This file originates from incubator-tvm

#include <dmlc/logging.h>
#include <mxnet/runtime/c_runtime_api.h>
#include <mxnet/runtime/object.h>
#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

#include "../c_api/c_api_common.h"
#include "./object_internal.h"

namespace mxnet {
namespace runtime {

/*! \brief Type information */
struct TypeInfo {
  /*! \brief The current index. */
  uint32_t index{0};
  /*! \brief Index of the parent in the type hierachy */
  uint32_t parent_index{0};
  // NOTE: the indices in [index, index + num_reserved_slots) are
  // reserved for the child-class of this type.
  /*! \brief Total number of slots reserved for the type and its children. */
  uint32_t num_slots{0};
  /*! \brief number of allocated child slots. */
  uint32_t allocated_slots{0};
  /*! \brief Whether child can overflow. */
  bool child_slots_can_overflow{true};
  /*! \brief name of the type. */
  std::string name;
  /*! \brief hash of the name */
  size_t name_hash{0};
};

/*!
 * \brief Type context that manages the type hierachy information.
 */
class TypeContext {
 public:
  // NOTE: this is a relatively slow path for child checking
  // Most types are already checked by the fast-path via reserved slot checking.
  bool DerivedFrom(uint32_t child_tindex, uint32_t parent_tindex) {
    // invariance: child's type index is always bigger than its parent.
    if (child_tindex < parent_tindex) return false;
    if (child_tindex == parent_tindex) return true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK_LT(child_tindex, type_table_.size());
      while (child_tindex > parent_tindex) {
        child_tindex = type_table_[child_tindex].parent_index;
      }
    }
    return child_tindex == parent_tindex;
  }

  uint32_t GetOrAllocRuntimeTypeIndex(const std::string& skey,
                                      uint32_t static_tindex,
                                      uint32_t parent_tindex,
                                      uint32_t num_child_slots,
                                      bool child_slots_can_overflow) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = type_key2index_.find(skey);
    if (it != type_key2index_.end()) {
      return it->second;
    }
    // try to allocate from parent's type table.
    CHECK_LT(parent_tindex, type_table_.size());
    TypeInfo& pinfo = type_table_[parent_tindex];
    CHECK_EQ(pinfo.index, parent_tindex);

    // if parent cannot overflow, then this class cannot.
    if (!pinfo.child_slots_can_overflow) {
      child_slots_can_overflow = false;
    }

    // total number of slots include the type itself.
    uint32_t num_slots = num_child_slots + 1;
    uint32_t allocated_tindex;

    if (static_tindex != TypeIndex::kDynamic) {
      // statically assigned type
      allocated_tindex = static_tindex;
      CHECK_LT(static_tindex, type_table_.size());
      CHECK_EQ(type_table_[allocated_tindex].allocated_slots, 0U)
          << "Conflicting static index " << static_tindex
          << " between " << type_table_[allocated_tindex].name
          << " and "
          << skey;
    } else if (pinfo.allocated_slots + num_slots < pinfo.num_slots) {
      // allocate the slot from parent's reserved pool
      allocated_tindex = parent_tindex + pinfo.allocated_slots;
      // update parent's state
      pinfo.allocated_slots += num_slots;
    } else {
      CHECK(pinfo.child_slots_can_overflow)
          << "Reach maximum number of sub-classes for " << pinfo.name;
      // allocate new entries.
      allocated_tindex = type_counter_;
      type_counter_ += num_slots;
      CHECK_LE(type_table_.size(), allocated_tindex);
      type_table_.resize(allocated_tindex + 1, TypeInfo());
    }
    CHECK_GT(allocated_tindex, parent_tindex);
    // initialize the slot.
    type_table_[allocated_tindex].index = allocated_tindex;
    type_table_[allocated_tindex].parent_index = parent_tindex;
    type_table_[allocated_tindex].num_slots = num_slots;
    type_table_[allocated_tindex].allocated_slots = 1;
    type_table_[allocated_tindex].child_slots_can_overflow =
        child_slots_can_overflow;
    type_table_[allocated_tindex].name = skey;
    type_table_[allocated_tindex].name_hash = std::hash<std::string>()(skey);
    // update the key2index mapping.
    type_key2index_[skey] = allocated_tindex;
    return allocated_tindex;
  }

  std::string TypeIndex2Key(uint32_t tindex) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(tindex < type_table_.size() &&
          type_table_[tindex].allocated_slots != 0)
        << "Unknown type index " << tindex;
    return type_table_[tindex].name;
  }

  size_t TypeIndex2KeyHash(uint32_t tindex) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(tindex < type_table_.size() &&
          type_table_[tindex].allocated_slots != 0)
        << "Unknown type index " << tindex;
    return type_table_[tindex].name_hash;
  }

  uint32_t TypeKey2Index(const std::string& skey) {
    auto it = type_key2index_.find(skey);
    CHECK(it != type_key2index_.end())
        << "Cannot find type " << skey;
    return it->second;
  }

  static TypeContext* Global() {
    static TypeContext inst;
    return &inst;
  }

 private:
  TypeContext() {
    type_table_.resize(TypeIndex::kStaticIndexEnd, TypeInfo());
  }
  // mutex to avoid registration from multiple threads.
  std::mutex mutex_;
  std::atomic<uint32_t> type_counter_{TypeIndex::kStaticIndexEnd};
  std::vector<TypeInfo> type_table_;
  std::unordered_map<std::string, uint32_t> type_key2index_;
};

uint32_t Object::GetOrAllocRuntimeTypeIndex(const std::string& key,
                                            uint32_t static_tindex,
                                            uint32_t parent_tindex,
                                            uint32_t num_child_slots,
                                            bool child_slots_can_overflow) {
  return TypeContext::Global()->GetOrAllocRuntimeTypeIndex(
      key, static_tindex, parent_tindex, num_child_slots, child_slots_can_overflow);
}

bool Object::DerivedFrom(uint32_t parent_tindex) const {
  return TypeContext::Global()->DerivedFrom(
      this->type_index_, parent_tindex);
}

std::string Object::TypeIndex2Key(uint32_t tindex) {
  return TypeContext::Global()->TypeIndex2Key(tindex);
}

size_t Object::TypeIndex2KeyHash(uint32_t tindex) {
  return TypeContext::Global()->TypeIndex2KeyHash(tindex);
}

uint32_t Object::TypeKey2Index(const std::string& key) {
  return TypeContext::Global()->TypeKey2Index(key);
}

}  // namespace runtime
}  // namespace mxnet

int MXNetObjectFree(MXNetObjectHandle obj) {
  API_BEGIN();
  mxnet::runtime::ObjectInternal::ObjectFree(obj);
  API_END();
}
