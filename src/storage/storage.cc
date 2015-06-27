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
  hd.dptr = new char[size];
  hd.ctx = ctx;
  hd.handle_ = NULL;
  return hd;
}
void NaiveStorageManager::Free(StorageManager::Handle handle) {
  char *dptr = static_cast<char*>(handle.dptr);
  delete [] dptr;
}
StorageManager *StorageManager::Get() {
  static NaiveStorageManager inst;
  return &inst;
}
}  // namespace mxnet
