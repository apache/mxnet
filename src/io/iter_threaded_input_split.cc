// Copyright by Contributors

#include <dmlc/io.h>
#include <dmlc/threadediter.h>
#include "./iter_threaded_input_split.h"

namespace mxnet {
namespace io {
dmlc::ThreadedIter<CommonChunkType>* CreateInputSplitThreadedIter(
    dmlc::InputSplit* base,
    size_t max_capacity,
    bool ChunkMode) {
  CHECK(base != nullptr);
  dmlc::ThreadedIter<CommonChunkType>* chunk_iter =
      new dmlc::ThreadedIter<CommonChunkType>(max_capacity);
  chunk_iter->Init([base, ChunkMode](CommonChunkType **dptr) {
                     dmlc::InputSplit::Blob blob;
                     bool base_end = ChunkMode ? (!base->NextChunk(&blob)) :
                         !base->NextRecord(&blob);
                     if (base_end) return false;
                     if (*dptr == nullptr) {
                       *dptr = new CommonChunkType();
                     }
                     (*dptr)->assign(static_cast<char*>(blob.dptr),
                                     static_cast<char*>(blob.dptr) + blob.size);
                     return true;
                   },
                   [base](void) { base->BeforeFirst(); });
  return chunk_iter;
}

}  // namespace io
}  // namespace mxnet
