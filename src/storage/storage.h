#pragma once
#include <cassert>
#include <map>
#include <memory>
#include "common/common.h"

typedef uint64_t StorageId;
typedef uint64_t DeviceId;

class StorageManager {
public:
  virtual StorageId makeNewInstance(size_t size) = 0;
  virtual StorageId makeNewInstance(void * memory, size_t size, DeviceId) = 0;
  virtual void * getStorage(StorageId, DeviceId) = 0;
  virtual void ref(StorageId) = 0;
  virtual void deref(StorageId) = 0;
};

class NaiveStorageManager : public StorageManager {
public:
  StorageId makeNewInstance(size_t size) override {
    blocks[currentId].size = size;
    return currentId++;
  }
  StorageId makeNewInstance(void * memory, size_t size, DeviceId did) override {
    auto itBoolPair = blocks.emplace(currentId, Storage());
    itBoolPair.first->second.size = size;
    auto data = itBoolPair.first->second.data;
    data[did] = memory;
    return currentId++;
  }
  void * getStorage(StorageId sid, DeviceId did) override {
    auto it = blocks.find(sid);
    assert(it != blocks.end());
    auto & data = it->second.data;
    auto itd = data.find(did);
    if (itd == data.end()) {
      // TODO: replace this part with Device code
      // no storage for this device, need to allocate
      void * newStore = new char[it->second.size];
      // then copy
      if (!data.empty()) {
        memcpy(newStore, data.begin()->second, it->second.size);
      }
      data.insert(itd, std::make_pair(did, newStore));
      return newStore;
    }
    else {
      return itd->second;
    }
  }
  virtual void ref(StorageId) {}
  virtual void deref(StorageId) {}
private:
  struct Storage {
    size_t size;
    std::map<DeviceId, void*> data;
  };
  StorageId currentId = 0;
  std::map<StorageId, Storage> blocks;
};