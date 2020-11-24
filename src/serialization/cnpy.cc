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

#include "zip.h"



namespace mxnet {

void fortran_order_transpose_prepare(std::vector<dim_t>& shape) {  // NOLINT(runtime/references)
  std::reverse(std::begin(shape), std::end(shape));
}

// NOLINTNEXTLINE(runtime/references)
NDArray fortran_order_transpose(std::vector<dim_t>& shape, int type_flag, NDArray& array) {
  std::reverse(std::begin(shape), std::end(shape));
  TShape tshape(shape);
  NDArray transposed(tshape, Context::CPU(), false, type_flag);
  const std::vector<NDArray*> inputs {&array};
  const std::vector<NDArray*> outputs {&transposed};
  const std::vector<OpReqType> reqs {kWriteTo};  // Transpose does not support kWriteInplace
  nnvm::NodeAttrs attrs;
  if (!Imperative::Get()->is_np_shape()) {
    attrs.op = nnvm::Op::Get("transpose");
  } else {
    attrs.op = nnvm::Op::Get("_npi_transpose");
  }
  attrs.op->attr_parser(&attrs);
  Imperative::Get()->InvokeOp(Context::CPU(), attrs, inputs, outputs,
                              reqs, DispatchMode::kFCompute, OpStatePtr());
  return transposed;
}


namespace npy {

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define MXNET_BYTEORDER "<"
#define MXNET_BYTEORDER_CHAR '<'
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define MXNET_BYTEORDER ">"
#define MXNET_BYTEORDER_CHAR '>'
#elif defined(_WIN32)
#define MXNET_BYTEORDER "<"
#define MXNET_BYTEORDER_CHAR '<'
#else
#error "endian detection needs to be set up for your compiler"
#endif

std::string dtype_descr(const TBlob& blob) {
  switch (blob.type_flag_) {
    case mshadow::kFloat16: return "'" MXNET_BYTEORDER "f2'";
    case mshadow::kFloat32: return "'" MXNET_BYTEORDER "f4'";
    case mshadow::kFloat64: return "'" MXNET_BYTEORDER "f8'";
    case mshadow::kInt8: return "'|i1'";
    case mshadow::kInt16: return "'" MXNET_BYTEORDER "i2'";
    case mshadow::kInt32: return "'" MXNET_BYTEORDER "i4'";
    case mshadow::kInt64: return "'" MXNET_BYTEORDER "i8'";
    case mshadow::kBool: return "'|b1'";
    case mshadow::kUint8: return "'|u1'";
    case mshadow::kUint16: return "'" MXNET_BYTEORDER "u2'";
    case mshadow::kUint32: return "'" MXNET_BYTEORDER "u4'";
    case mshadow::kUint64: return "'" MXNET_BYTEORDER "u8'";
    case mshadow::kBfloat16: return "[('bfloat16', '" MXNET_BYTEORDER "u2')]";
    default: {
      LOG(FATAL) << "Unknown dtype type " << blob.type_flag_ << "encountered.";
      return "";
    }
  }
}


int dtype_descr(const std::string& dtype_descr) {
    if (dtype_descr.find("f2'") != std::string::npos) return mshadow::kFloat16;
    else if (dtype_descr.find("f4'") != std::string::npos) return mshadow::kFloat32;
    else if (dtype_descr.find("f8'") != std::string::npos) return mshadow::kFloat64;
    else if (dtype_descr.find("|i1'") != std::string::npos) return mshadow::kInt8;
    else if (dtype_descr.find("i2'") != std::string::npos) return mshadow::kInt16;
    else if (dtype_descr.find("i4'") != std::string::npos) return mshadow::kInt32;
    else if (dtype_descr.find("i8'") != std::string::npos) return mshadow::kInt64;
    else if (dtype_descr.find("|b1'") != std::string::npos) return mshadow::kBool;
    else if (dtype_descr.find("|u1'") != std::string::npos) return mshadow::kUint8;
    else if (dtype_descr.find("u2'") != std::string::npos) return mshadow::kUint16;
    else if (dtype_descr.find("u4'") != std::string::npos) return mshadow::kUint32;
    else if (dtype_descr.find("u8'") != std::string::npos) return mshadow::kUint64;
    else if (dtype_descr.find("bfloat16'") != std::string::npos) return mshadow::kBfloat16;
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
  loc = header.find('(');
  std::string::size_type end_loc = header.find(')');
  CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: '('";
  CHECK_NE(end_loc, std::string::npos) << "failed to find NPY header keyword: ')'";
  std::string shape_str = header.substr(loc+1, end_loc-loc-1);
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
#if MXNET_USE_MKLDNN == 1
    if (array.IsMKLDNNData()) {
      array = array.Reorder2Default();
    }
#endif
  }

  CHECK_EQ(array.storage_type(), kDefaultStorage);

  const TBlob& blob = array.data();
  std::string npy_header = create_npy_header(blob);

  std::ofstream output(fname, std::ios::binary);
  output.write(npy_header.data(), npy_header.size());
  output.write(static_cast<const char*>(blob.dptr_), blob.Size() *
               mshadow::mshadow_sizeof(blob.type_flag_));
}

NDArray load_array(const std::string& fname) {
  std::ifstream strm(fname, std::ios::binary);
  strm.exceptions(std::istream::eofbit);
  strm.exceptions(std::istream::failbit);
  strm.exceptions(std::istream::badbit);

  uint32_t header_len = parse_npy_header_len(strm);
  std::string header(header_len, ' ');
  strm.read(header.data(), header_len);
  auto[type_flag, fortran_order, shape] = parse_npy_header_descr(header);

  if (fortran_order) {
    fortran_order_transpose_prepare(shape);
  }

  TShape tshape(shape);
  NDArray array(tshape, Context::CPU(), false, type_flag);
  const TBlob& blob = array.data();
  strm.read(reinterpret_cast<char*>(blob.dptr_), blob.Size() *
            mshadow::mshadow_sizeof(blob.type_flag_));

  if (fortran_order) {
    array = fortran_order_transpose(shape, type_flag, array);
  }

  return array;
}

}  // namespace npy

namespace npz {


void save_blob(int zip_open_flags, const std::string& zip_fname, const std::string& blob_name,
               const TBlob& blob) {
  int error;
  zip_t* archive = zip_open(zip_fname.c_str(), zip_open_flags, &error);
  if (archive == nullptr) {
    zip_error_t e;
    zip_error_init_with_code(&e, error);
    throw std::runtime_error(zip_error_strerror(&e));
  }

  std::string npy_header = npy::create_npy_header(blob);

  // Declare buffers from making up the .npy file
  std::array<zip_buffer_fragment_t, 2> fragments;
  fragments[0].data = reinterpret_cast<zip_uint8_t*>(npy_header.data());
  fragments[0].length = npy_header.size();
  fragments[1].data = reinterpret_cast<zip_uint8_t*>(blob.dptr_);
  fragments[1].length = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);

  zip_error_t e;
  zip_source_t* source =
      zip_source_buffer_fragment_create(fragments.data(), fragments.size(), 0, &e);
  if (source == nullptr) {
      throw std::runtime_error(zip_error_strerror(&e));
  }
  zip_int64_t index = zip_file_add(archive, (blob_name + ".npy").data(), source, ZIP_FL_ENC_UTF_8);
  if (index < 0) {
    zip_source_free(source);
    throw std::runtime_error(zip_strerror(archive));
  }
  error = zip_set_file_compression(archive, index, ZIP_CM_STORE, 0);
  if (error != 0) {
      std::string strerror{zip_strerror(archive)};
      zip_discard(archive);
      throw std::runtime_error(strerror);
  }

  // Write everything
  error = zip_close(archive);
  if (error != 0) {
    std::string strerror{zip_strerror(archive)};
    zip_discard(archive);
    throw std::runtime_error(strerror);
  }
}


// Save shape of sparse ndarray in to scipy compatible shape.npy with int64 data
void save_shape_array(int zip_open_flags, const std::string& zip_fname,
                      const std::string& blob_name, const mxnet::TShape& shape) {
  int error;
  zip_t* archive = zip_open(zip_fname.c_str(), zip_open_flags, &error);
  if (archive == nullptr) {
    zip_error_t e;
    zip_error_init_with_code(&e, error);
    throw std::runtime_error(zip_error_strerror(&e));
  }

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

  zip_error_t e;
  zip_source_t* source = zip_source_buffer_create(npy.data(), npy.size(), 0, &e);
  if (source == nullptr) {
      throw std::runtime_error(zip_error_strerror(&e));
  }
  zip_int64_t index = zip_file_add(archive, (blob_name + ".npy").data(), source, ZIP_FL_ENC_UTF_8);
  if (index < 0) {
    zip_source_free(source);
    throw std::runtime_error(zip_strerror(archive));
  }
  error = zip_set_file_compression(archive, index, ZIP_CM_STORE, 0);
  if (error != 0) {
      std::string strerror{zip_strerror(archive)};
      zip_discard(archive);
      throw std::runtime_error(strerror);
  }

  // Write everything
  error = zip_close(archive);
  if (error != 0) {
    std::string strerror{zip_strerror(archive)};
    zip_discard(archive);
    throw std::runtime_error(strerror);
  }
}


void save_format_array(int zip_open_flags, const std::string& zip_fname,
                       const std::string& blob_name, const std::string_view& format) {
  int error;
  zip_t* archive = zip_open(zip_fname.c_str(), zip_open_flags, &error);
  if (archive == nullptr) {
    zip_error_t e;
    zip_error_init_with_code(&e, error);
    throw std::runtime_error(zip_error_strerror(&e));
  }

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

  zip_error_t e;
  zip_source_t* source = zip_source_buffer_create(npy.data(), npy.size(), 0, &e);
  if (source == nullptr) {
      throw std::runtime_error(zip_error_strerror(&e));
  }
  zip_int64_t index = zip_file_add(archive, (blob_name + ".npy").data(), source, ZIP_FL_ENC_UTF_8);
  if (index < 0) {
    zip_source_free(source);
    throw std::runtime_error(zip_strerror(archive));
  }
  error = zip_set_file_compression(archive, index, ZIP_CM_STORE, 0);
  if (error != 0) {
      std::string strerror{zip_strerror(archive)};
      zip_discard(archive);
      throw std::runtime_error(strerror);
  }

  // Write everything
  error = zip_close(archive);
  if (error != 0) {
    std::string strerror{zip_strerror(archive)};
    zip_discard(archive);
    throw std::runtime_error(strerror);
  }
}


void save_array(int write_mode, const std::string& zip_fname, const std::string& array_name,
                const NDArray& array_) {
  NDArray array;  // a copy on cpu
  if (array_.ctx().dev_mask() != cpu::kDevMask) {
    array = array_.Copy(Context::CPU());
    array.WaitToRead();
  } else {
    array = array_;
    array.WaitToRead();
#if MXNET_USE_MKLDNN == 1
    if (array.IsMKLDNNData()) {
      array = array.Reorder2Default();
    }
#endif
  }

  switch (array.storage_type()) {
  case kDefaultStorage: {
    save_blob(write_mode, zip_fname, array_name, array.data());
    break;
  }
  case kCSRStorage: {
    save_blob(write_mode, zip_fname, array_name + "/data", array.data());
    write_mode = 0;  // Append to the created zip file going forward
    save_blob(write_mode, zip_fname, array_name + "/indptr", array.aux_data(csr::kIndPtr));
    save_blob(write_mode, zip_fname, array_name + "/indices", array.aux_data(csr::kIdx));
    save_shape_array(write_mode, zip_fname, array_name + "/shape", array.shape());
    save_format_array(write_mode, zip_fname, array_name + "/format", "csr");
    break;
  }
  case kRowSparseStorage: {
    save_blob(write_mode, zip_fname, array_name + "/data", array.data());
    write_mode = 0;  // Append to the created zip file going forward
    save_blob(write_mode, zip_fname, array_name + "/indices", array.aux_data(rowsparse::kIdx));
    save_shape_array(write_mode, zip_fname, array_name + "/shape", array.shape());
    save_format_array(write_mode, zip_fname, array_name + "/format", "row_sparse");
    break;
  }
  default: LOG(FATAL) << "Unknown storage type " << array.storage_type() << "encountered.";
  }
}


uint32_t parse_npy_header_len(zip_file_t* file, const std::string_view& fname,
                              const std::string& zip_fname) {
  std::array<char, 12> buffer;
  zip_int64_t bytesread = zip_fread(file, buffer.data(), 10);
  if (bytesread != 10) {
    LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
  }
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
    zip_int64_t bytesread = zip_fread(file, &buffer[10], 2);
    if (bytesread != 2) {
      LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
    }
    header_len += buffer[10] >> 16;
    header_len += buffer[11] >> 24;
  }
  return header_len;
}


std::pair<std::vector<NDArray>, std::vector<std::string>>
load_arrays(const std::string& zip_fname) {
  int error;
  zip_t* archive = zip_open(zip_fname.c_str(), ZIP_RDONLY, &error);
  if (archive == nullptr) {
    zip_error_t e;
    zip_error_init_with_code(&e, error);
    throw std::runtime_error(zip_error_strerror(&e));
  }

  // Collect the set of file-names per folder in the zip file. If the set of
  // file names in a folder matches the scipy.sparse.save_npz pattern, the
  // folder will be restored as single sparse ndarray.
  std::unordered_map<std::string_view, std::set<std::string_view>> names;

  zip_int64_t num_entries = zip_get_num_entries(archive, ZIP_FL_UNCHANGED);
  for (zip_uint64_t i = 0; i < num_entries; i++) {
    std::string_view entry_name = zip_get_name(archive, i, ZIP_FL_ENC_STRICT);
    if (entry_name.substr(entry_name.size() - 4).compare(".npy") != 0) continue;  // only .npy

    auto dir_sep_search = entry_name.rfind("/");
    if (dir_sep_search == std::string::npos) {  // top level file
      [[maybe_unused]] auto[iter, inserted] = names[""].insert(entry_name);
      CHECK(inserted);
    } else {  // file inside a folder
      std::string_view dirname = entry_name.substr(0, dir_sep_search + 1);
      std::string_view fname = entry_name.substr(dir_sep_search + 1);
      [[maybe_unused]] auto[iter, inserted] = names[dirname].insert(fname);
      CHECK(inserted);
    }
  }

  // Return values
  std::vector<NDArray> arrays;
  std::vector<std::string> return_names;

  // Patterns used by SciPy to save respective sparse matrix formats to a file
  const std::set<std::string_view> bsr_csr_csc_pattern
    {"data.npy", "indices.npy", "indptr.npy", "format.npy", "shape.npy"};
  const std::set<std::string_view> row_sparse_pattern  // MXNet specific format not part of SciPy
    {"data.npy", "indices.npy", "format.npy", "shape.npy"};
  const std::set<std::string_view> coo_pattern
    {"data.npy", "row.npy", "col.npy", "format.npy", "shape.npy"};
  const std::set<std::string_view> dia_pattern
    {"data.npy", "offsets.npy", "format.npy", "shape.npy"};
  for (const auto& [dirname, dircontents] : names) {
    if (dircontents == bsr_csr_csc_pattern) {
      // Check format
      std::string fname(dirname);
      fname += "format.npy";
      zip_file_t* format_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
      if (format_file == nullptr) {
        throw std::runtime_error(zip_strerror(archive));
      }

      // In the special case of format.npy we ignore the header as it
      // specifies the string datatype which is unsupported by MXNet
      uint32_t header_len = parse_npy_header_len(format_file, fname, zip_fname);
      std::string header;
      header.resize(header_len);
      zip_int64_t bytesread = zip_fread(format_file, header.data(), header_len);
      if (bytesread != header_len) {
        LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
      }
      // and simply look at the next 3 bytes containing the format string
      std::string format;
      format.resize(3);
      bytesread = zip_fread(format_file, format.data(), 3);
      zip_fclose(format_file);
      if (bytesread != 3) {
        LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
      }

      if (format == "csr") {
        // Prepare reading storage data array
        fname = dirname;
        fname += "data.npy";
        zip_file_t* data_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (data_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        uint32_t header_len = parse_npy_header_len(data_file, fname, zip_fname);
        std::string header;
        header.resize(header_len);
        zip_int64_t bytesread = zip_fread(data_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[storage_type_flag, storage_fortran_order, storage_shape] = \
          npy::parse_npy_header_descr(header);
        if (storage_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape storage_tshape(storage_shape);

        // Prepare reading indptr aux array
        fname = dirname;
        fname += "indptr.npy";
        zip_file_t* indptr_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (indptr_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        header_len = parse_npy_header_len(indptr_file, fname, zip_fname);
        header.resize(header_len);
        bytesread = zip_fread(indptr_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[indptr_type_flag, indptr_fortran_order, indptr_shape] = \
          npy::parse_npy_header_descr(header);
        if (indptr_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indptr_tshape(indptr_shape);

        // Prepare reading indices aux array
        fname = dirname;
        fname += "indices.npy";
        zip_file_t* indices_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (indices_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        header_len = parse_npy_header_len(indices_file, fname, zip_fname);
        header.resize(header_len);
        bytesread = zip_fread(indices_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[indices_type_flag, indices_fortran_order, indices_shape] = \
          npy::parse_npy_header_descr(header);
        if (indices_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indices_tshape(indices_shape);

        // Read shape data array
        fname = dirname;
        fname += "shape.npy";
        zip_file_t* shape_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (shape_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        header_len = parse_npy_header_len(shape_file, fname, zip_fname);
        header.resize(header_len);
        bytesread = zip_fread(shape_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[shape_type_flag, shape_fortran_order, shape_shape] = \
          npy::parse_npy_header_descr(header);
        if (shape_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        CHECK_EQ(shape_shape.size(), 1) << "Expected one-dimensional shape of shape information.";
        TShape tshape(shape_shape.at(0), -1);
        if (shape_type_flag == mshadow::kInt64) {  // Used in most SciPy builds
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int64_t dim;
            bytesread = zip_fread(shape_file, &dim, 8);
            if (bytesread != 8) {
              LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
            }
            tshape[i] = dim;
          }
        } else if (shape_type_flag == mshadow::kInt32) {  // Used in SciPy pip wheels on Windows
          for (dim_t i = 0; i < shape_shape.at(0); i++) {
            int32_t dim;
            bytesread = zip_fread(shape_file, &dim, 4);
            if (bytesread != 4) {
              LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
            }
            tshape[i] = dim;
          }
        } else {
          LOG(FATAL) << "Expected shape information in int64 or int32 format.";
        }
        zip_fclose(shape_file);

        // Construct aux datastructures
        static_assert(csr::CSRAuxType::kIndPtr == 0);
        static_assert(csr::CSRAuxType::kIdx == 1);
        const std::vector<int> aux_types {indptr_type_flag, indices_type_flag};
        const mxnet::ShapeVector aux_shapes {indptr_tshape, indices_tshape};

        // Allocate NDArray
        NDArray array(NDArrayStorageType::kCSRStorage, tshape, Context::CPU(), false,
                      storage_type_flag, aux_types, aux_shapes, storage_tshape);

        // Read data array
        const TBlob& blob = array.data();
        zip_uint64_t nbytes = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
        bytesread = zip_fread(data_file, blob.dptr_, nbytes);
        zip_fclose(data_file);
        if (bytesread != nbytes) {
          LOG(FATAL) << "Failed to read from data.npy member of " << zip_fname;
        }

        // Read indptr array
        const TBlob& indptr_blob = array.aux_data(csr::CSRAuxType::kIndPtr);
        nbytes = indptr_blob.Size() * mshadow::mshadow_sizeof(indptr_blob.type_flag_);
        bytesread = zip_fread(indptr_file, indptr_blob.dptr_, nbytes);
        zip_fclose(indptr_file);
        if (bytesread != nbytes) {
          LOG(FATAL) << "Failed to read from indptr.npy member of " << zip_fname;
        }

        // Read indices array
        const TBlob& indices_blob = array.aux_data(csr::CSRAuxType::kIdx);
        nbytes = indices_blob.Size() * mshadow::mshadow_sizeof(indices_blob.type_flag_);
        bytesread = zip_fread(indices_file, indices_blob.dptr_, nbytes);
        zip_fclose(indices_file);
        if (bytesread != nbytes) {
          LOG(FATAL) << "Failed to read from indices.npy member of " << zip_fname;
        }

        arrays.push_back(array);
        return_names.emplace_back(dirname.size() ?   // Exclude "/"
                                  dirname.substr(0, dirname.size() - 1) : dirname);

      } else {
        throw std::runtime_error("Loading " + format + " sparse matrix format is unsupported.");
      }
    } else if (dircontents == row_sparse_pattern) {
      // Check format
      std::string fname(dirname);
      fname += "format.npy";
      zip_file_t* format_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
      if (format_file == nullptr) {
        throw std::runtime_error(zip_strerror(archive));
      }

      // In the special case of format.npy we ignore the header as it
      // specifies the string datatype which is unsupported by MXNet
      uint32_t header_len = parse_npy_header_len(format_file, fname, zip_fname);
      std::string header;
      header.resize(header_len);
      zip_int64_t bytesread = zip_fread(format_file, header.data(), header_len);
      if (bytesread != header_len) {
        LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
      }
      // and simply look at the next 10 bytes containing the format string
      std::string format;
      format.resize(10);
      bytesread = zip_fread(format_file, format.data(), 10);
      zip_fclose(format_file);

      if (format == "row_sparse") {
        // Prepare reading storage data array
        fname = dirname;
        fname += "data.npy";
        zip_file_t* data_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (data_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        uint32_t header_len = parse_npy_header_len(data_file, fname, zip_fname);
        std::string header;
        header.resize(header_len);
        zip_int64_t bytesread = zip_fread(data_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[storage_type_flag, storage_fortran_order, storage_shape] = \
          npy::parse_npy_header_descr(header);
        if (storage_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape storage_tshape(storage_shape);

        // Prepare reading indices aux array
        fname = dirname;
        fname += "indices.npy";
        zip_file_t* indices_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (indices_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        header_len = parse_npy_header_len(indices_file, fname, zip_fname);
        header.resize(header_len);
        bytesread = zip_fread(indices_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[indices_type_flag, indices_fortran_order, indices_shape] = \
          npy::parse_npy_header_descr(header);
        if (indices_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        TShape indices_tshape(indices_shape);

        // Read shape data array
        fname = dirname;
        fname += "shape.npy";
        zip_file_t* shape_file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (shape_file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }
        header_len = parse_npy_header_len(shape_file, fname, zip_fname);
        header.resize(header_len);
        bytesread = zip_fread(shape_file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto[shape_type_flag, shape_fortran_order, shape_shape] = \
          npy::parse_npy_header_descr(header);
        if (shape_fortran_order) {
          LOG(FATAL) << "Reading fortran order data for sparse arrays not yet implemented.";
        }
        CHECK_EQ(shape_type_flag, mshadow::kInt64) << "Expected shape information in int64 format.";
        CHECK_EQ(shape_shape.size(), 1) << "Expected one-dimensional shape of shape information.";
        TShape tshape(shape_shape.at(0), -1);
        for (dim_t i = 0; i < shape_shape.at(0); i++) {
          int64_t dim;
          bytesread = zip_fread(shape_file, &dim, 8);
          if (bytesread != 8) {
            LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
          }
          tshape[i] = dim;
        }
        zip_fclose(shape_file);

        // Construct aux datastructures
        static_assert(rowsparse::RowSparseAuxType::kIdx == 0);
        const std::vector<int> aux_types {indices_type_flag};
        const mxnet::ShapeVector aux_shapes {indices_tshape};

        // Allocate NDArray
        NDArray array(NDArrayStorageType::kRowSparseStorage, tshape, Context::CPU(), false,
                      storage_type_flag, aux_types, aux_shapes, storage_tshape);

        // Read data array
        const TBlob& blob = array.data();
        zip_uint64_t nbytes = blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_);
        bytesread = zip_fread(data_file, blob.dptr_, nbytes);
        zip_fclose(data_file);
        if (bytesread != nbytes) {
          LOG(FATAL) << "Failed to read from data.npy member of " << zip_fname;
        }

        // Read indices array
        const TBlob& indices_blob = array.aux_data(rowsparse::RowSparseAuxType::kIdx);
        nbytes = indices_blob.Size() * mshadow::mshadow_sizeof(indices_blob.type_flag_);
        bytesread = zip_fread(indices_file, indices_blob.dptr_, nbytes);
        zip_fclose(indices_file);
        if (bytesread != nbytes) {
          LOG(FATAL) << "Failed to read from indices.npy member of " << zip_fname;
        }

        arrays.push_back(array);
        return_names.emplace_back(dirname.size() ?   // Exclude "/"
                                  dirname.substr(0, dirname.size() - 1) : dirname);

      } else {
        throw std::runtime_error("Loading " + format + " sparse matrix format is unsupported.");
      }
    } else if (dircontents == coo_pattern) {
      throw std::runtime_error("Loading COO sparse matrix format is unsupported.");
    } else if (dircontents == dia_pattern) {
      throw std::runtime_error("Loading DIA sparse matrix format is unsupported.");
    } else {  // Folder does not match scipy sparse pattern; treat containing files as dense
      for (const std::string_view& fname : dircontents) {
        std::string path(dirname);
        path += fname;

        // The string_view points to a null-terminated character array
        // owned by zip_get_name and thus conversion to C char* is valid
        zip_file_t* file = zip_fopen(archive, path.data(), ZIP_FL_UNCHANGED);
        if (file == nullptr) {
          throw std::runtime_error(zip_strerror(archive));
        }

        uint32_t header_len = parse_npy_header_len(file, path, zip_fname);
        std::string header;
        header.resize(header_len);
        zip_int64_t bytesread = zip_fread(file, header.data(), header_len);
        if (bytesread != header_len) {
          LOG(FATAL) << "Failed to read from " << path << " member of " << zip_fname;
        }
        auto[type_flag, fortran_order, shape] = npy::parse_npy_header_descr(header);

        if (fortran_order) {
          fortran_order_transpose_prepare(shape);
        }

        TShape tshape(shape);
        NDArray array(tshape, Context::CPU(), false, type_flag);
        const TBlob& blob = array.data();
        bytesread = zip_fread(file, blob.dptr_,
                              blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));
        zip_fclose(file);
        if (bytesread != blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_)) {
          LOG(FATAL) << "Failed to read from " << path << " member of " << zip_fname;
        }

        if (fortran_order) {
          array = fortran_order_transpose(shape, type_flag, array);
        }

        arrays.push_back(array);
        return_names.emplace_back(path.substr(0, path.size() - 4));  // Skip .npy
      }
    }
  }

  zip_discard(archive);

  return std::make_pair(arrays, return_names);
}

}  // namespace npz
}  // namespace mxnet
