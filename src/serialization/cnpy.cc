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

#include "cnpy.h"
#include <stdint.h>
#include <fstream>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>
#include "zip.h"

namespace mxnet {

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
    else LOG(FATAL) << "Unknown dtype descriptor " << dtype_descr << "encountered.";
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
    header += (char)0x93;
    header += "NUMPY";

    std::string::size_type size = dict.size();
    CHECK(size <= std::numeric_limits<uint32_t>::max()) << "Shape too large for NPY serialization";
    if (size <= std::numeric_limits<uint16_t>::max()) {
        header += (char)0x01;  //major version of numpy format
        header += (char)0x00;  //minor version of numpy format
        uint16_t size_ = dict.size();
        header += (char) (size_ & 0xFF);
        header += (char) (size_ >> 8);
    } else {
        header += (char)0x02;  //major version of numpy format
        header += (char)0x00;  //minor version of numpy format
        uint32_t size_ = dict.size();
        header += (char) (size_ & 0xFF);
        header += (char) ((size_ >> 8) & 0xFF);
        header += (char) ((size_ >> 16) & 0xFF);
        header += (char) ((size_ >> 24) & 0xFF);
    }

    header += dict;

    return header;
}

uint32_t parse_npy_header_len(std::ifstream& strm) {
    strm.exceptions(std::istream::eofbit);
    strm.exceptions(std::istream::failbit);
    strm.exceptions(std::istream::badbit);

    CHECK(strm.get() == 0x93);
    CHECK(strm.get() == 'N');
    CHECK(strm.get() == 'U');
    CHECK(strm.get() == 'M');
    CHECK(strm.get() == 'P');
    CHECK(strm.get() == 'Y');

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

std::pair<int, mxnet::TShape> parse_npy_header_descr(const std::string& header) {
    // Fortran order
    std::string::size_type loc = header.find("fortran_order");
    CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: 'fortran_order'";
    bool fortran_order = (header.substr(loc + 16, 4) == "True" ? true : false);
    // TODO: support fortran order
    CHECK_EQ(fortran_order, false) << "Loading files in fortran_order is not yet supported. "
                                   << "Please open the file with numpy.load and convert "
                                   << "to non-fortran order";

    // Shape
    loc = header.find("(");
    std::string::size_type end_loc = header.find(")");
    CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: '('";
    CHECK_NE(end_loc, std::string::npos) << "failed to find NPY header keyword: ')'";
    std::string shape_str = header.substr(loc+1, end_loc-loc-1);
    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    std::vector<dim_t> shape_;
    while(std::regex_search(shape_str, sm, num_regex)) {
        shape_.push_back(std::stoi(sm[0].str()));
        shape_str = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    loc = header.find("descr");
    CHECK_NE(loc, std::string::npos) << "failed to find NPY header keyword: 'descr'";
    // May use https://github.com/numpy/numpy/blob/38275835/numpy/core/src/multiarray/ctors.c#L365
    CHECK_EQ(header[loc + 9], MXNET_BYTEORDER_CHAR) << "Loading files with non-native endianness "
                                                    << "is not yet supported. Please open the file "
                                                    << "with numpy.load, use byteswap method to "
                                                    << "convert endianness and re-save the file.";

    int type_flag = dtype_descr(header);
    mxnet::TShape shape(shape_);
    return std::pair(type_flag, shape);
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
  output.write(static_cast<const char*>( blob.dptr_ ), blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));
}

NDArray load_array(const std::string& fname) {
    std::ifstream strm(fname, std::ios::binary);
    strm.exceptions(std::istream::eofbit);
    strm.exceptions(std::istream::failbit);
    strm.exceptions(std::istream::badbit);

    uint32_t header_len = parse_npy_header_len(strm);
    std::string header(header_len, ' ');
    strm.read(header.data(), header_len);
    auto [type_flag, shape] = parse_npy_header_descr(header);

    NDArray array(shape, Context::CPU(), false, type_flag);
    const TBlob& blob = array.data();
    strm.read(reinterpret_cast<char*>(blob.dptr_), blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));
    return array;
}

}

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
      // TODO shape array
      // TODO format ("csr")
      break;
    }
    case kRowSparseStorage: {
        save_blob(write_mode, zip_fname, array_name + "/data", array.data());
      write_mode = 0;  // Append to the created zip file going forward
      save_blob(write_mode, zip_fname, array_name + "/indices", array.aux_data(rowsparse::kIdx));
      // TODO shape array
      // TODO format ("rowsparse")
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
    CHECK(buffer[0] == (char)0x93);
    CHECK(buffer[1] == 'N');
    CHECK(buffer[2] == 'U');
    CHECK(buffer[3] == 'M');
    CHECK(buffer[4] == 'P');
    CHECK(buffer[5] == 'Y');
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


std::pair<std::vector<NDArray>, std::vector<std::string>> load_arrays(const std::string& zip_fname) {

    int error;
    zip_t* archive = zip_open(zip_fname.c_str(), ZIP_RDONLY, &error);
    if (archive == nullptr) {
        zip_error_t e;
        zip_error_init_with_code(&e, error);
        throw std::runtime_error(zip_error_strerror(&e));
    }

    zip_int64_t num_entries = zip_get_num_entries(archive, ZIP_FL_UNCHANGED);
    std::vector<std::string_view> dense_names;  // root-level files
    std::unordered_map<std::string_view, std::vector<std::string_view>> sparse_names;
    for (zip_uint64_t i = 0; i < num_entries; i++) {
        std::string_view entry_name = zip_get_name(archive, i, ZIP_FL_ENC_STRICT);
        // only consider .npy files
        if (entry_name.substr(entry_name.size() - 4).compare(".npy") != 0) continue;
        auto dir_sep_search = entry_name.rfind("/");
        if (dir_sep_search != std::string::npos) {
            // Array in a folder; may be part of sparse ndarray
            std::string_view fname = entry_name.substr(dir_sep_search + 1);
            if (fname == "data.npy" || fname == "indices.npy" || fname == "indptr.npy") {
                // Filename matches special names used by scipy.sparse.save_npz
                // Use parent path as name of sparse array
                std::string_view sparse_name = fname.substr(0, dir_sep_search);
                // Record filenames in the directory to later decide the type of sparse array
                sparse_names[sparse_name].push_back(fname);
            } else {
                dense_names.push_back(entry_name.substr(0, entry_name.size() - 4));
            }
        } else {
            dense_names.push_back(entry_name.substr(0, entry_name.size() - 4));
        }
    }

    // Return values
    std::vector<NDArray> arrays;
    std::vector<std::string> names;

    // Handle dense arrays
    for (const std::string_view& fname : dense_names) {
        // fname string_view refers to null-termiated character array owned by zip_get_name
        zip_file_t* file = zip_fopen(archive, fname.data(), ZIP_FL_UNCHANGED);
        if (file == nullptr) {
            throw std::runtime_error(zip_strerror(archive));
        }

        uint32_t header_len = parse_npy_header_len(file, fname, zip_fname);
        std::string header;
        header.resize(header_len);
        zip_int64_t bytesread = zip_fread(file, header.data(), header_len);
        if (bytesread != header_len) {
            LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }
        auto [type_flag, shape] = npy::parse_npy_header_descr(header);

        NDArray array(shape, Context::CPU(), false, type_flag);
        const TBlob& blob = array.data();
        bytesread = zip_fread(file, blob.dptr_, blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_));
        if (bytesread != blob.Size() * mshadow::mshadow_sizeof(blob.type_flag_)) {
            LOG(FATAL) << "Failed to read from " << fname << " member of " << zip_fname;
        }

        arrays.push_back(array);
        names.emplace_back(fname);
    }

    // TODO Handle sparse arrays
    CHECK_EQ(sparse_names.size(), 0);

    return std::make_pair(arrays, names);
}

}  // namespace npz
}  // namespace mxnet
