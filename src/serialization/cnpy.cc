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

// File is based on https://github.com/leezu/cnpy/tree/libzip released under MIT License
// Copyright (C) 2011  Carl Rogers, 2018 Leonard Lausen

#include "cnpy.h"
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include <string_view>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <complex>
#include <numeric>
#include <limits>
#include <regex>
#include <tuple>
#include <set>
#include <stdexcept>
#include <typeinfo>

namespace mxnet {

void fortran_order_transpose_prepare(std::vector<dim_t>& shape) {  // NOLINT(runtime/references)
  std::reverse(std::begin(shape), std::end(shape));
}

// NOLINTNEXTLINE(runtime/references)
NDArray fortran_order_transpose(std::vector<dim_t>& shape, int type_flag, NDArray& array) {
  std::reverse(std::begin(shape), std::end(shape));
  TShape tshape(shape);
  NDArray transposed(tshape, Context::CPU(), false, type_flag);
  const std::vector<NDArray*> inputs{&array};
  const std::vector<NDArray*> outputs{&transposed};
  const std::vector<OpReqType> reqs{kWriteTo};  // Transpose does not support kWriteInplace
  nnvm::NodeAttrs attrs;
  if (!Imperative::Get()->is_np_shape()) {
    attrs.op = nnvm::Op::Get("transpose");
  } else {
    attrs.op = nnvm::Op::Get("_npi_transpose");
  }
  attrs.op->attr_parser(&attrs);
  Imperative::Get()->InvokeOp(
      Context::CPU(), attrs, inputs, outputs, reqs, DispatchMode::kFCompute, OpStatePtr());
  return transposed;
}

namespace npy {

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define MXNET_BYTEORDER      "<"
#define MXNET_BYTEORDER_CHAR '<'
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define MXNET_BYTEORDER      ">"
#define MXNET_BYTEORDER_CHAR '>'
#elif defined(_WIN32)
#define MXNET_BYTEORDER      "<"
#define MXNET_BYTEORDER_CHAR '<'
#else
#error "endian detection needs to be set up for your compiler"
#endif

std::string dtype_descr(const TBlob& blob) {
  switch (blob.type_flag_) {
    case mshadow::kFloat16:
      return "'" MXNET_BYTEORDER "f2'";
    case mshadow::kFloat32:
      return "'" MXNET_BYTEORDER "f4'";
    case mshadow::kFloat64:
      return "'" MXNET_BYTEORDER "f8'";
    case mshadow::kInt8:
      return "'|i1'";
    case mshadow::kInt16:
      return "'" MXNET_BYTEORDER "i2'";
    case mshadow::kInt32:
      return "'" MXNET_BYTEORDER "i4'";
    case mshadow::kInt64:
      return "'" MXNET_BYTEORDER "i8'";
    case mshadow::kBool:
      return "'|b1'";
    case mshadow::kUint8:
      return "'|u1'";
    case mshadow::kUint16:
      return "'" MXNET_BYTEORDER "u2'";
    case mshadow::kUint32:
      return "'" MXNET_BYTEORDER "u4'";
    case mshadow::kUint64:
      return "'" MXNET_BYTEORDER "u8'";
    case mshadow::kBfloat16:
      return "'" MXNET_BYTEORDER "b2'";
    default: {
      LOG(FATAL) << "Unknown dtype type " << blob.type_flag_ << "encountered.";
      return "";
    }
  }
}

int dtype_descr(const std::string& dtype_descr) {
  if (dtype_descr.find("f2'") != std::string::npos)
    return mshadow::kFloat16;
  else if (dtype_descr.find("f4'") != std::string::npos)
    return mshadow::kFloat32;
  else if (dtype_descr.find("f8'") != std::string::npos)
    return mshadow::kFloat64;
  else if (dtype_descr.find("|i1'") != std::string::npos)
    return mshadow::kInt8;
  else if (dtype_descr.find("i2'") != std::string::npos)
    return mshadow::kInt16;
  else if (dtype_descr.find("i4'") != std::string::npos)
    return mshadow::kInt32;
  else if (dtype_descr.find("i8'") != std::string::npos)
    return mshadow::kInt64;
  else if (dtype_descr.find("|b1'") != std::string::npos)
    return mshadow::kBool;
  else if (dtype_descr.find("|u1'") != std::string::npos)
    return mshadow::kUint8;
  else if (dtype_descr.find("u2'") != std::string::npos)
    return mshadow::kUint16;
  else if (dtype_descr.find("u4'") != std::string::npos)
    return mshadow::kUint32;
  else if (dtype_descr.find("u8'") != std::string::npos)
    return mshadow::kUint64;
  else if (dtype_descr.find("b2'") != std::string::npos)
    return mshadow::kBfloat16;
  else
    LOG(FATAL) << "Unknown dtype descriptor " << dtype_descr << "encountered.";
  return -1;
}

std::string create_npy_header(const TBlob& blob) {
  std::string dict;
  dict += "{'descr': ";
  dict += dtype_descr(blob);
  dict += ", 'fortran_order': False, 'shape': (";
  if (blob.ndim()) {
    dict += std::to_string(blob.shape_[0]);
    for (int i = 1; i < blob.ndim(); i++) {
      dict += ", ";
      dict += std::to_string(blob.shape_[i]);
    }
    if (blob.ndim() == 1) {
      dict += ",";
    }
  }
  dict += "), }";

  // pad with spaces so that preamble+dict is modulo 64 bytes. preamble is
  // 10 bytes. dict needs to end with \n
  int remainder = 64 - (10 + dict.size() + 1) % 64;
  dict.insert(dict.end(), remainder, ' ');
  dict.push_back('\n');
  assert((dict.size() + 10) % 64 == 0);

  std::string header;
  header += static_cast<char>(0x93);
  header += "NUMPY";

  std::string::size_type size = dict.size();
  CHECK(size <= std::numeric_limits<uint32_t>::max()) << "Shape too large for NPY serialization";
  if (size <= std::numeric_limits<uint16_t>::max()) {
    header += static_cast<char>(0x01);  // major version of numpy format
    header += static_cast<char>(0x00);  // minor version of numpy format
    uint16_t size_ = dict.size();
    header += static_cast<char>(size_ & 0xFF);
    header += static_cast<char>(size_ >> 8);
  } else {
    header += static_cast<char>(0x02);  // major version of numpy format
    header += static_cast<char>(0x00);  // minor version of numpy format
    uint32_t size_ = dict.size();
    header += static_cast<char>(size_ & 0xFF);
    header += static_cast<char>((size_ >> 8) & 0xFF);
    header += static_cast<char>((size_ >> 16) & 0xFF);
    header += static_cast<char>((size_ >> 24) & 0xFF);
  }

  header += dict;

  return header;
}

uint32_t parse_npy_header_len(std::ifstream& strm) {
  strm.exceptions(std::istream::eofbit);
  strm.exceptions(std::istream::failbit);
  strm.exceptions(std::istream::badbit);

  CHECK_EQ(strm.get(), 0x93);
  CHECK_EQ(strm.get(), 'N');
  CHECK_EQ(strm.get(), 'U');
  CHECK_EQ(strm.get(), 'M');
  CHECK_EQ(strm.get(), 'P');
  CHECK_EQ(strm.get(), 'Y');

  uint8_t major_version = strm.get();
  CHECK(major_version == 0x01 || major_version == 0x02) << "Unsupported npy major version";
  CHECK(strm.get() == 0x00) << "Unsupported npy minor version";

  uint32_t header_len = 0;
  header_len += strm.get();
  header_len += strm.get() >> 8;
  if (major_version == 0x02) {
    header_len += strm.get() >> 16;
    header_len += strm.get() >> 24;
  }
  return header_len;
}

std::tuple<int, int, std::vector<dim_t>> parse_npy_header_descr(const std::string& header) {
  // Fortran order
  std::string::size_type loc = header.find("fortran_order");
  CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: 'fortran_order'";
  bool fortran_order = (header.substr(loc + 16, 4) == "True" ? true : false);

  // Shape
  loc                            = header.find('(');
  std::string::size_type end_loc = header.find(')');
  CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: '('";
  CHECK_NE(end_loc, std::string::npos) << "failed to find NPY header keyword: ')'";
  std::string shape_str = header.substr(loc + 1, end_loc - loc - 1);
  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  std::vector<dim_t> shape;
  while (std::regex_search(shape_str, sm, num_regex)) {
    shape.push_back(std::stoi(sm[0].str()));
    shape_str = sm.suffix().str();
  }

  // endian, word size, data type
  // byte order code | stands for not applicable.
  loc = header.find("descr");
  CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: 'descr'";
  // May use https://github.com/numpy/numpy/blob/38275835/numpy/core/src/multiarray/ctors.c#L365
  CHECK(header[loc + 9] == MXNET_BYTEORDER_CHAR || header[loc + 9] == '|')
      << "Loading files with non-native endianness "
      << "is not yet supported. Please open the file "
      << "with numpy.load, use byteswap method to "
      << "convert endianness and re-save the file.";

  int type_flag = dtype_descr(header);
  return std::tuple(type_flag, fortran_order, shape);
}

void save_array(const std::string& fname, const NDArray& array_) {
  NDArray array;  // a copy on cpu
  if (array_.ctx().dev_mask() != cpu::kDevMask) {
    array = array_.Copy(Context::CPU());
    array.WaitToRead();
  } else {
    array = array_;
    array.WaitToRead();
#if MXNET_USE_ONEDNN == 1
    if (array.IsDNNLData()) {
      array = array.Reorder2Default();
    }
#endif
  }

  CHECK_EQ(array.storage_type(), kDefaultStorage);

  const TBlob& blob      = array.data();
  std::string npy_header = create_npy_header(blob);

  std::ofstream output(fname, std::ios::binary);
  output.write(npy_header.data(), npy_header.size());
  output.write(static_cast<const char*>(blob.dptr_),
               blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));
}

NDArray load_array(const std::string& fname) {
  std::ifstream strm(fname, std::ios::binary);
  strm.exceptions(std::istream::eofbit);
  strm.exceptions(std::istream::failbit);
  strm.exceptions(std::istream::badbit);

  uint32_t header_len = parse_npy_header_len(strm);
  std::string header(header_len, ' ');
  strm.read(header.data(), header_len);
  auto [type_flag, fortran_order, shape] = parse_npy_header_descr(header);  // NOLINT

  if (fortran_order) {
    fortran_order_transpose_prepare(shape);
  }

  TShape tshape(shape);
  NDArray array(tshape, Context::CPU(), false, type_flag);
  const TBlob& blob = array.data();
  strm.read(reinterpret_cast<char*>(blob.dptr_),
            blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));

  if (fortran_order) {
    array = fortran_order_transpose(shape, type_flag, array);
  }

  return array;
}

}  // namespace npy

namespace npz {

size_t npy_header_blob_read_callback(void* pOpaque, mz_uint64 file_ofs, void* pBuf, size_t n) {
  auto [npy_header, blob] =                                                  // NOLINT
      *static_cast<std::tuple<const std::string*, const TBlob*>*>(pOpaque);  // NOLINT

  if (file_ofs < npy_header->size() && file_ofs + n < npy_header->size()) {
    // Read n bytes from npy_header
    const void* pSrc = static_cast<const void*>(npy_header->data() + file_ofs);
    std::memcpy(pBuf, pSrc, n);
  } else if (file_ofs < npy_header->size()) {
    // Read npy_header->size() - file_ofs bytes from npy_header
    const void* pSrc          = static_cast<const void*>(npy_header->data() + file_ofs);
    const size_t npy_header_n = npy_header->size() - file_ofs;
    std::memcpy(pBuf, pSrc, npy_header_n);

    // Read n - (npy_header->size() - file_ofs) bytes from blob
    void* pBuf_blob = static_cast<void*>(static_cast<char*>(pBuf) + npy_header_n);
    std::memcpy(pBuf_blob, blob->dptr_, n - npy_header_n);
  } else {
    // Read n bytes from blob
    const void* pSrc =
        static_cast<const void*>(static_cast<char*>(blob->dptr_) + file_ofs - npy_header->size());
    std::memcpy(pBuf, pSrc, n);
  }
  return n;
}

void save_blob(mz_zip_archive* archive, const std::string& blob_name, const TBlob& blob) {
  const std::string npy_header = npy::create_npy_header(blob);

  const std::string blob_name_npy = blob_name + ".npy";
  mz_uint64 size_to_add           = npy_header.size();
  size_to_add += blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
  auto callback_data = std::tuple(&npy_header, &blob);
  CHECK(mz_zip_writer_add_read_buf_callback(archive,
                                            blob_name_npy.data(),
                                            npy_header_blob_read_callback,
                                            static_cast<void*>(&callback_data),
                                            size_to_add,
                                            nullptr,
                                            nullptr,
                                            0,
                                            MZ_NO_COMPRESSION,
                                            nullptr,
                                            0,
                                            nullptr,
                                            0))
      << mz_zip_get_error_string(mz_zip_get_last_error(archive));
}

// Save shape of sparse ndarray in to scipy compatible shape.npy with int64 data
void save_shape_array(mz_zip_archive* archive,
                      const std::string& blob_name,
                      const mxnet::TShape& shape) {
  // Special case of create_npy_header for TShape data
  std::string dict;
  dict += "{'descr': '<i8', 'fortran_order': False, 'shape': (";
  dict += std::to_string(shape.ndim());
  dict += ",), }";
  // pad with spaces so that preamble+dict is modulo 64 bytes. preamble is
  // 10 bytes. dict needs to end with \n
  int remainder = 64 - (10 + dict.size() + 1) % 64;
  dict.insert(dict.end(), remainder, ' ');
  dict.push_back('\n');
  assert((dict.size() + 10) % 64 == 0);
  std::string npy;
  npy += static_cast<char>(0x93);
  npy += "NUMPY";
  std::string::size_type size = dict.size();
  CHECK(size <= std::numeric_limits<uint32_t>::max()) << "Shape too large for NPY serialization";
  if (size <= std::numeric_limits<uint16_t>::max()) {
    npy += static_cast<char>(0x01);  // major version of numpy format
    npy += static_cast<char>(0x00);  // minor version of numpy format
    uint16_t size_ = dict.size();
    npy += static_cast<char>(size_ & 0xFF);
    npy += static_cast<char>(size_ >> 8);
  } else {
    npy += static_cast<char>(0x02);  // major version of numpy format
    npy += static_cast<char>(0x00);  // minor version of numpy format
    uint32_t size_ = dict.size();
    npy += static_cast<char>(size_ & 0xFF);
    npy += static_cast<char>((size_ >> 8) & 0xFF);
    npy += static_cast<char>((size_ >> 16) & 0xFF);
    npy += static_cast<char>((size_ >> 24) & 0xFF);
  }
  npy += dict;

  // Add shape data
  for (const uint64_t value : shape) {
    npy += static_cast<char>(value & 0xFF);
    npy += static_cast<char>((value >> 8) & 0xFF);
    npy += static_cast<char>((value >> 16) & 0xFF);
    npy += static_cast<char>((value >> 24) & 0xFF);
    npy += static_cast<char>((value >> 32) & 0xFF);
    npy += static_cast<char>((value >> 40) & 0xFF);
    npy += static_cast<char>((value >> 48) & 0xFF);
    npy += static_cast<char>((value >> 56) & 0xFF);
  }

  const std::string blob_name_npy = blob_name + ".npy";
  CHECK(mz_zip_writer_add_mem(
      archive, blob_name_npy.data(), npy.data(), npy.size(), MZ_NO_COMPRESSION))
      << mz_zip_get_error_string(mz_zip_get_last_error(archive));
}

void save_format_array(mz_zip_archive* archive,
                       const std::string& blob_name,
                       const std::string_view& format) {
  // Special case of create_npy_header for TShape data
  std::string dict;
  dict += "{'descr': '|s";
  dict += std::to_string(format.size());
  dict += "{'descr': '<i8', 'fortran_order': False, 'shape': (), }";
  // pad with spaces so that preamble+dict is modulo 64 bytes. preamble is
  // 10 bytes. dict needs to end with \n
  int remainder = 64 - (10 + dict.size() + 1) % 64;
  dict.insert(dict.end(), remainder, ' ');
  dict.push_back('\n');
  assert((dict.size() + 10) % 64 == 0);
  std::string npy;
  npy += static_cast<char>(0x93);
  npy += "NUMPY";
  std::string::size_type size = dict.size();
  CHECK(size <= std::numeric_limits<uint32_t>::max());
  if (size <= std::numeric_limits<uint16_t>::max()) {
    npy += static_cast<char>(0x01);  // major version of numpy format
    npy += static_cast<char>(0x00);  // minor version of numpy format
    uint16_t size_ = dict.size();
    npy += static_cast<char>(size_ & 0xFF);
    npy += static_cast<char>(size_ >> 8);
  } else {
    npy += static_cast<char>(0x02);  // major version of numpy format
    npy += static_cast<char>(0x00);  // minor version of numpy format
    uint32_t size_ = dict.size();
    npy += static_cast<char>(size_ & 0xFF);
    npy += static_cast<char>((size_ >> 8) & 0xFF);
    npy += static_cast<char>((size_ >> 16) & 0xFF);
    npy += static_cast<char>((size_ >> 24) & 0xFF);
  }
  npy += dict;

  npy += format;

  const std::string blob_name_npy = blob_name + ".npy";
  CHECK(mz_zip_writer_add_mem(
      archive, blob_name_npy.data(), npy.data(), npy.size(), MZ_NO_COMPRESSION))
      << mz_zip_get_error_string(mz_zip_get_last_error(archive));
}

void save_array(mz_zip_archive* archive, const std::string& array_name, const NDArray& array_) {
  NDArray array;  // a copy on cpu
  if (array_.ctx().dev_mask() != cpu::kDevMask) {
    array = array_.Copy(Context::CPU());
    array.WaitToRead();
  } else {
    array = array_;
    array.WaitToRead();
#if MXNET_USE_ONEDNN == 1
    if (array.IsDNNLData()) {
      array = array.Reorder2Default();
    }
#endif
  }

  switch (array.storage_type()) {
    case kDefaultStorage: {
      save_blob(archive, array_name, array.data());
      break;
    }
    case kCSRStorage: {
      save_blob(archive, array_name + "/data", array.data());
      save_blob(archive, array_name + "/indptr", array.aux_data(csr::kIndPtr));
      save_blob(archive, array_name + "/indices", array.aux_data(csr::kIdx));
      save_shape_array(archive, array_name + "/shape", array.shape());
      save_format_array(archive, array_name + "/format", "csr");
      break;
    }
    case kRowSparseStorage: {
      save_blob(archive, array_name + "/data", array.data());
      save_blob(archive, array_name + "/indices", array.aux_data(rowsparse::kIdx));
      save_shape_array(archive, array_name + "/shape", array.shape());
      save_format_array(archive, array_name + "/format", "row_sparse");
      break;
    }
    default:
      LOG(FATAL) << "Unknown storage type " << array.storage_type() << "encountered.";
  }
}

uint32_t parse_npy_header_len(mz_zip_reader_extract_iter_state* state,
                              const std::string_view& fname,
                              const std::string& zip_fname) {
  std::array<char, 12> buffer;
  CHECK_EQ(mz_zip_reader_extract_iter_read(state, buffer.data(), 10), 10)
      << "Failed to read from " << fname << " member of " << zip_fname;
  CHECK_EQ(buffer[0], (char)0x93);
  CHECK_EQ(buffer[1], 'N');
  CHECK_EQ(buffer[2], 'U');
  CHECK_EQ(buffer[3], 'M');
  CHECK_EQ(buffer[4], 'P');
  CHECK_EQ(buffer[5], 'Y');
  uint8_t major_version = buffer[6];
  CHECK(major_version == 0x01 || major_version == 0x02) << "Unsupported npy major version";
  CHECK(buffer[7] == 0x00) << "Unsupported npy minor version";
  uint32_t header_len = 0;
  header_len += buffer[8];
  header_len += buffer[9] >> 8;
  if (major_version == 0x02) {
    CHECK_EQ(mz_zip_reader_extract_iter_read(state, &buffer[10], 2), 2)
        << "Failed to read from " << fname << " member of " << zip_fname;
    header_len += buffer[10] >> 16;
    header_len += buffer[11] >> 24;
  }
  return header_len;
}

std::pair<std::vector<NDArray>, std::vector<std::string>> load_arrays(
    const std::string& zip_fname) {
  mz_zip_archive archive{};
  CHECK(mz_zip_reader_init_file(&archive, zip_fname.data(), 0))
      << "Failed to open archive " << zip_fname << ": "
      << mz_zip_get_error_string(mz_zip_get_last_error(&archive));

  // Collect the set of file-names per folder in the zip file. If the set of
  // file names in a folder matches the scipy.sparse.save_npz pattern, the
  // folder will be restored as single sparse ndarray.
  std::unordered_map<std::string, std::set<std::string>> names;

  mz_uint num_entries = mz_zip_reader_get_num_files(&archive);
  for (mz_uint i = 0; i < num_entries; i++) {
    mz_uint filename_length = mz_zip_reader_get_filename(&archive, i, nullptr, 0);
    std::string entry_name;
    entry_name.resize(filename_length);  // filename_length includes the \0 terminator
    CHECK_EQ(filename_length,
             mz_zip_reader_get_filename(&archive, i, entry_name.data(), filename_length));
    std::string_view entry_name_v{entry_name.data(), entry_name.size() - 1};  // -1 due to \0
    if (entry_name_v.substr(entry_name_v.size() - 4).compare(".npy") != 0)
      continue;  // only .npy

    auto dir_sep_search = entry_name_v.rfind("/");
    if (dir_sep_search == std::string::npos) {                                   // top level file
      [[maybe_unused]] auto [iter, inserted] = names[""].emplace(entry_name_v);  // NOLINT
      CHECK(inserted);
    } else {  // file inside a folder
      std::string dirname{entry_name_v.substr(0, dir_sep_search + 1)};
      std::string fname{entry_name_v.substr(dir_sep_search + 1)};
      [[maybe_unused]] auto [iter, inserted] = names[dirname].insert(fname);  // NOLINT
      CHECK(inserted);
    }
  }

  // Return values
  std::vector<NDArray> arrays;
  std::vector<std::string> return_names;

  // Patterns used by SciPy to save respective sparse matrix formats to a file
  const std::set<std::string> bsr_csr_csc_pattern{
      "data.npy", "indices.npy", "indptr.npy", "format.npy", "shape.npy"};
  const std::set<std::string> row_sparse_pattern  // MXNet specific format not part of SciPy
      {"data.npy", "indices.npy", "format.npy", "shape.npy"};
  const std::set<std::string> coo_pattern{
      "data.npy", "row.npy", "col.npy", "format.npy", "shape.npy"};
  const std::set<std::string> dia_pattern{"data.npy", "offsets.npy", "format.npy", "shape.npy"};
  for (const auto& [dirname, dircontents] : names) {
    if (dircontents == bsr_csr_csc_pattern) {
      // Check format
      std::string fname(dirname);
      fname += "format.npy";
      mz_zip_reader_extract_iter_state* format_file =
          mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
      CHECK(nullptr != format_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));

      // In the special case of format.npy we ignore the header as it
      // specifies the string datatype which is unsupported by MXNet
      uint32_t header_len = parse_npy_header_len(format_file, fname, zip_fname);
      std::string header;
      header.resize(header_len);
      CHECK_EQ(mz_zip_reader_extract_iter_read(format_file, header.data(), header_len), header_len)
          << "Failed to read from " << fname << " member of " << zip_fname << ": "
          << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
      // and simply look at the next 3 bytes containing the format string
      std::string format;
      format.resize(3);
      CHECK_EQ(mz_zip_reader_extract_iter_read(format_file, format.data(), 3), 3)
          << "Failed to read from " << fname << " member of " << zip_fname;
      CHECK(mz_zip_reader_extract_iter_free(format_file));

      if (format == "csr") {
        // Prepare reading storage data array
        fname = dirname;
        fname += "data.npy";
        mz_zip_reader_extract_iter_state* data_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != data_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(data_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(data_file, header.data(), header_len), header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [storage_type_flag, storage_fortran_order, storage_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (storage_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape storage_tshape(storage_shape);

        // Prepare reading indptr aux array
        fname = dirname;
        fname += "indptr.npy";
        mz_zip_reader_extract_iter_state* indptr_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != indptr_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(indptr_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indptr_file, header.data(), header_len),
                 header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [indptr_type_flag, indptr_fortran_order, indptr_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (indptr_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indptr_tshape(indptr_shape);

        // Prepare reading indices aux array
        fname = dirname;
        fname += "indices.npy";
        mz_zip_reader_extract_iter_state* indices_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != indices_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(indices_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indices_file, header.data(), header_len),
                 header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [indices_type_flag, indices_fortran_order, indices_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (indices_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indices_tshape(indices_shape);

        // Read shape data array
        fname = dirname;
        fname += "shape.npy";
        mz_zip_reader_extract_iter_state* shape_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != shape_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(shape_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, header.data(), header_len), header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [shape_type_flag, shape_fortran_order, shape_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (shape_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        CHECK_EQ(shape_shape.size(), 1) << "Expected one-dimensional shape of shape information.";
        TShape tshape(shape_shape.at(0), -1);
        if (shape_type_flag == mshadow::kInt64) {  // Used in most SciPy builds
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int64_t dim;
            CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, &dim, 8), 8)
                << "Failed to read from " << fname << " member of " << zip_fname << ": "
                << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
            tshape[i] = dim;
          }
        } else if (shape_type_flag == mshadow::kInt32) {  // Used in SciPy pip wheels on Windows
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int32_t dim;
            CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, &dim, 4), 4)
                << "Failed to read from " << fname << " member of " << zip_fname << ": "
                << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
            tshape[i] = dim;
          }
        } else {
          LOG(FATAL) << "Expected shape information in int64 or int32 format.";
        }
        CHECK(mz_zip_reader_extract_iter_free(shape_file));

        // Construct aux datastructures
        static_assert(csr::CSRAuxType::kIndPtr == 0);
        static_assert(csr::CSRAuxType::kIdx == 1);
        const std::vector<int> aux_types{indptr_type_flag, indices_type_flag};
        const mxnet::ShapeVector aux_shapes{indptr_tshape, indices_tshape};

        // Allocate NDArray
        NDArray array(NDArrayStorageType::kCSRStorage,
                      tshape,
                      Context::CPU(),
                      false,
                      storage_type_flag,
                      aux_types,
                      aux_shapes,
                      storage_tshape);

        // Read data array
        const TBlob& blob = array.data();
        size_t nbytes     = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(data_file, blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(data_file));

        // Read indptr array
        const TBlob& indptr_blob = array.aux_data(csr::CSRAuxType::kIndPtr);
        nbytes = indptr_blob.Size() * mshadow::mshadow_sizeof(indptr_blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indptr_file, indptr_blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(indptr_file));

        // Read indices array
        const TBlob& indices_blob = array.aux_data(csr::CSRAuxType::kIdx);
        nbytes = indices_blob.Size() * mshadow::mshadow_sizeof(indices_blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indices_file, indices_blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(indices_file));

        arrays.push_back(array);
        return_names.emplace_back(dirname.size() ?  // Exclude "/"
                                      dirname.substr(0, dirname.size() - 1) :
                                      dirname);

      } else {
        throw std::runtime_error("Loading " + format + " sparse matrix format is unsupported.");
      }
    } else if (dircontents == row_sparse_pattern) {
      // Check format
      std::string fname(dirname);
      fname += "format.npy";
      mz_zip_reader_extract_iter_state* format_file =
          mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
      CHECK(nullptr != format_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));

      // In the special case of format.npy we ignore the header as it
      // specifies the string datatype which is unsupported by MXNet
      uint32_t header_len = parse_npy_header_len(format_file, fname, zip_fname);
      std::string header;
      header.resize(header_len);
      CHECK_EQ(mz_zip_reader_extract_iter_read(format_file, header.data(), header_len), header_len)
          << "Failed to read from " << fname << " member of " << zip_fname << ": "
          << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
      // and simply look at the next 3 bytes containing the format string
      std::string format;
      format.resize(10);
      mz_zip_reader_extract_iter_read(format_file, format.data(), 10);
      CHECK(mz_zip_reader_extract_iter_free(format_file));

      if (format == "row_sparse") {
        // Prepare reading storage data array
        fname = dirname;
        fname += "data.npy";
        mz_zip_reader_extract_iter_state* data_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != data_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(data_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(data_file, header.data(), header_len), header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [storage_type_flag, storage_fortran_order, storage_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (storage_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape storage_tshape(storage_shape);

        // Prepare reading indices aux array
        fname = dirname;
        fname += "indices.npy";
        mz_zip_reader_extract_iter_state* indices_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != indices_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(indices_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indices_file, header.data(), header_len),
                 header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [indices_type_flag, indices_fortran_order, indices_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (indices_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indices_tshape(indices_shape);

        // Read shape data array
        fname = dirname;
        fname += "shape.npy";
        mz_zip_reader_extract_iter_state* shape_file =
            mz_zip_reader_extract_file_iter_new(&archive, fname.data(), 0);
        CHECK(nullptr != shape_file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        header_len = parse_npy_header_len(shape_file, fname, zip_fname);
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, header.data(), header_len), header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [shape_type_flag, shape_fortran_order, shape_shape] =  // NOLINT
            npy::parse_npy_header_descr(header);
        if (shape_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        CHECK_EQ(shape_shape.size(), 1) << "Expected one-dimensional shape of shape information.";
        TShape tshape(shape_shape.at(0), -1);
        if (shape_type_flag == mshadow::kInt64) {  // Used in most SciPy builds
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int64_t dim;
            CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, &dim, 8), 8)
                << "Failed to read from " << fname << " member of " << zip_fname << ": "
                << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
            tshape[i] = dim;
          }
        } else if (shape_type_flag == mshadow::kInt32) {  // Used in SciPy pip wheels on Windows
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int32_t dim;
            CHECK_EQ(mz_zip_reader_extract_iter_read(shape_file, &dim, 4), 4)
                << "Failed to read from " << fname << " member of " << zip_fname << ": "
                << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
            tshape[i] = dim;
          }
        } else {
          LOG(FATAL) << "Expected shape information in int64 or int32 format.";
        }
        CHECK(mz_zip_reader_extract_iter_free(shape_file));

        // Construct aux datastructures
        static_assert(rowsparse::RowSparseAuxType::kIdx == 0);
        const std::vector<int> aux_types{indices_type_flag};
        const mxnet::ShapeVector aux_shapes{indices_tshape};

        // Allocate NDArray
        NDArray array(NDArrayStorageType::kRowSparseStorage,
                      tshape,
                      Context::CPU(),
                      false,
                      storage_type_flag,
                      aux_types,
                      aux_shapes,
                      storage_tshape);

        // Read data array
        const TBlob& blob = array.data();
        size_t nbytes     = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(data_file, blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(data_file));

        // Read indices array
        const TBlob& indices_blob = array.aux_data(rowsparse::RowSparseAuxType::kIdx);
        nbytes = indices_blob.Size() * mshadow::mshadow_sizeof(indices_blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(indices_file, indices_blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(indices_file));

        arrays.push_back(array);
        return_names.emplace_back(dirname.size() ?  // Exclude "/"
                                      dirname.substr(0, dirname.size() - 1) :
                                      dirname);

      } else {
        throw std::runtime_error("Loading " + format + " sparse matrix format is unsupported.");
      }
    } else if (dircontents == coo_pattern) {
      throw std::runtime_error("Loading COO sparse matrix format is unsupported.");
    } else if (dircontents == dia_pattern) {
      throw std::runtime_error("Loading DIA sparse matrix format is unsupported.");
    } else {  // Folder does not match scipy sparse pattern; treat containing files as dense
      for (const std::string& fname : dircontents) {
        std::string path(dirname);
        path += fname;
        mz_zip_reader_extract_iter_state* file =
            mz_zip_reader_extract_file_iter_new(&archive, path.data(), 0);
        CHECK(nullptr != file) << mz_zip_get_error_string(mz_zip_get_last_error(&archive));

        uint32_t header_len = parse_npy_header_len(file, path, zip_fname);
        std::string header;
        header.resize(header_len);
        CHECK_EQ(mz_zip_reader_extract_iter_read(file, header.data(), header_len), header_len)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        auto [type_flag, fortran_order, shape] = npy::parse_npy_header_descr(header);  // NOLINT

        if (fortran_order) {
          fortran_order_transpose_prepare(shape);
        }

        TShape tshape(shape);
        NDArray array(tshape, Context::CPU(), false, type_flag);
        const TBlob& blob = array.data();
        size_t nbytes     = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
        CHECK_EQ(mz_zip_reader_extract_iter_read(file, blob.dptr_, nbytes), nbytes)
            << "Failed to read from " << fname << " member of " << zip_fname << ": "
            << mz_zip_get_error_string(mz_zip_get_last_error(&archive));
        CHECK(mz_zip_reader_extract_iter_free(file));

        if (fortran_order) {
          array = fortran_order_transpose(shape, type_flag, array);
        }

        arrays.push_back(array);
        return_names.emplace_back(path.substr(0, path.size() - 4));
      }
    }
  }

  mz_zip_reader_end(&archive);

  return std::make_pair(arrays, return_names);
}

}  // namespace npz
}  // namespace mxnet
