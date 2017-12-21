#ifndef MXNET_KVSTORE_UTILS_H_
#define MXNET_KVSTORE_UTILS_H_

#include <dmlc/logging.h>
#include <mxnet/ndarray.h>
#include <mxnet/resource.h>
#include <utility>
#include <vector>

namespace mxnet {
namespace kvstore {

bool CheckSameRowid(
    const std::vector<std::pair<NDArray*, NDArray>>& val_rowids) {
  MSHADOW_TYPE_SWITCH(val_rowids[0].second.dtype(), IType, {
    const TBlob& rowid_first = val_rowids[0].second.data();
    const IType *first_dptr = rowid_first.dptr<IType>();
    const index_t first_size = rowid_first.Size();
    for (size_t i = 1; i < val_rowids.size(); ++i) {
      const TBlob& rowid_i = val_rowids[i].second.data();
      if (rowid_i.dptr<IType>() != first_dptr
          || rowid_i.Size() != first_size) {
        return false;
      }
    }
  });
  return true;
}


}
}
#endif  // MXNET_KVSTORE_UTILS_H_
