/*!
 *  Copyright (c) 2015 by Contributors
 * \file global.cc
 * \brief Implementation of project global related functions.
 */
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/storage.h>
#include <mxnet/resource.h>
#include <mxnet/kvstore.h>

namespace mxnet {
// finalize the mxnet modules
void Finalize() {
  ResourceManager::Get()->Finalize();
  KVStore::Get()->Finalize();
  Engine::Get()->WaitForAll();
  Engine::Get()->Finalize();
  Storage::Get()->Finalize();
}
}  // namespace mxnet
