/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol_registry.cc
 * \brief symbol_registry of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/atomic_symbol.h>
#include <mxnet/atomic_symbol_registry.h>
#include <iterator>

namespace mxnet {

AtomicSymbolRegistry *AtomicSymbolRegistry::Get() {
  static AtomicSymbolRegistry instance;
  return &instance;
}

}  // namespace mxnet
