/*!
 *  Copyright (c) 2017 by shuqian.qu@hobot.cc
 * \file iter_recordio.cc
 * \brief record iter
 */
#include "./iter_recordio.h"

#if PB_FORMAT_REC
// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::RecordIteratorReg);
}  // namespace dmlc

namespace mxnet {
namespace io {
DMLC_REGISTER_PARAMETER(RecParserParam);
DMLC_REGISTER_PARAMETER(RecordDataInstParam);

MXNET_REGISTER_RECORD_ITER(RecordIter)
.describe(R"code(Iterating on RecordIO pb-format files

Read one record from RecordIO files with a rich of data load options.

One can use ``tools/im2rec.py --pb-format=1`` to pack individual files
into RecordIO files.

)code" ADD_FILELINE)
.add_arguments(RecParserParam::__FIELDS__())
.add_arguments(RecordDataInstParam::__FIELDS__())
.set_body([]() {
    return new RecordIter();
  });

}  // namespace io
}  // namespace mxnet

#endif  // PB_FORMAT_REC
