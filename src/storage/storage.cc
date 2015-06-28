#include <mshadow/tensor.h>
#include <mxnet/storage.h>
namespace mxnet {
class NaiveStorageManager : public StorageManager {
 public:
  virtual Handle Alloc(size_t size, Context ctx);
  virtual void Free(Handle handle);  
};

StorageManager::Handle
NaiveStorageManager::Alloc(size_t size, Context ctx) {
  Handle hd;
  hd.ctx = ctx;
  hd.handle_ = NULL;  
  if (ctx.dev_mask == cpu::kDevMask) {
    cudaMallocHost(&hd.dptr, size);    
  } else {
#if MXNET_USE_CUDA
    cudaMalloc(&hd.dptr, size);
#endif
  }
  return hd;
}
void NaiveStorageManager::Free(StorageManager::Handle handle) {
  if (handle.ctx.dev_mask == cpu::kDevMask) {
    cudaFreeHost(handle.dptr);
  } else {
#if MXNET_USE_CUDA
    cudaFree(handle.dptr);
#endif
  }
}
StorageManager *StorageManager::Get() {
  static NaiveStorageManager inst;
  return &inst;
}
}  // namespace mxnet
