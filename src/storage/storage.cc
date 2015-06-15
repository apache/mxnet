#include <mxnet/storage.h>
namespace mxnet {
class NaiveStorageManager : public StorageManager {
 public:
  virtual Handle Alloc(size_t size, int dev_mask, int dev_id);
  virtual void Free(Handle handle);  
};

StorageManager::Handle
NaiveStorageManager::Alloc(size_t size, int dev_mask, int dev_id) {
  Handle hd;
  hd.dptr = new char[size];
  hd.dev_mask = dev_mask;
  hd.dev_id = dev_id;
  hd.handle_ = NULL;
  return hd;
}
void NaiveStorageManager::Free(StorageManager::Handle handle) {
  char *dptr = static_cast<char*>(handle.dptr);
  delete [] dptr;
}
}  // namespace mxnet
