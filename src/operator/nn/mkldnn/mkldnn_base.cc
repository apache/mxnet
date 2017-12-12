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

#include "./mkldnn_base-inl.h"

namespace mxnet {

mkldnn::memory *TmpMemMgr::Alloc(const mkldnn::memory::primitive_desc &pd) {
  // We need to include the size of the memory used for alignment.
  this->est_size += pd.get_size() + alignment;
  void *this_mem = this->curr_mem;
  void *mem = std::align(alignment, pd.get_size(), this_mem, this->curr_size);
  if (mem) {
    // The memory is allocated from the temporary memory space in the
    // operator. It'll only become invalid after we exit from the operator.
    mkldnn_mem_ptr ret(new mkldnn::memory(pd, this_mem));
    MKLDNNStream::Instance().RegisterMem(ret);
    CHECK_EQ(this_mem, mem);
    this->curr_size -= pd.get_size();
    this->curr_mem = static_cast<char *>(this_mem) + pd.get_size();
    return ret.get();
  } else {
    LOG(WARNING) << "Allocate " << pd.get_size()
        << " bytes with malloc directly";
    mkldnn_mem_ptr ret(new mkldnn::memory(pd));
    MKLDNNStream::Instance().RegisterMem(ret);
    return ret.get();
  }
}

}  // namespace mxnet
