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

/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_recordio.h
 * \brief image recordio struct
 */
#ifndef MXNET_IO_IMAGE_RECORDIO_H_
#define MXNET_IO_IMAGE_RECORDIO_H_

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <string>

namespace mxnet {
namespace io {
/*! \brief image recordio struct */
struct ImageRecordIO {
  /*! \brief header in image recordio */
  struct Header {
    /*!
     * \brief flag of the header,
     *  used for future extension purposes
     */
    uint32_t flag;
    /*!
     * \brief label field that returns label of images
     *  when image list was not presented,
     *
     * NOTE: user do not need to repack recordio just to
     * change label field, just supply a list file that
     * maps image id to new labels
     */
    float label;
    /*!
     * \brief unique image index
     *  image_id[1] is always set to 0,
     *  reserved for future purposes for 128bit id
     *  image_id[0] is used to store image id
     */
    uint64_t image_id[2];
  };
  /*! \brief header of image recordio */
  Header header;
  /*! \brief point to label */
  float *label;
  /*! \brief number of float labels */
  int num_label;
  /*! \brief pointer to data content */
  uint8_t *content;
  /*! \brief size of the content */
  size_t content_size;
  /*! \brief constructor */
  ImageRecordIO(void)
      : label(nullptr), num_label(0), content(nullptr), content_size(0) {
    memset(&header, 0, sizeof(header));
  }
  /*! \brief get image id from record */
  inline uint64_t image_index(void) const {
    return header.image_id[0];
  }
  /*!
   * \brief load header from a record content
   * \param buf the head of record
   * \param size the size of the entire record
   */
  inline void Load(void *buf, size_t size) {
    CHECK(size >= sizeof(header));
    std::memcpy(&header, buf, sizeof(header));
    content = reinterpret_cast<uint8_t*>(buf) + sizeof(header);
    content_size = size - sizeof(header);
    if (header.flag > 0) {
      CHECK(content_size >= sizeof(float)*header.flag);
      label = reinterpret_cast<float*>(content);
      num_label = header.flag;
      content = reinterpret_cast<uint8_t*>(label + header.flag);
      content_size -= sizeof(float)*header.flag;
    } else {
      label = nullptr;
      num_label = 0;
    }
  }
  /*!
   * \brief save the record header
   */
  inline void SaveHeader(std::string *blob) const {
    blob->resize(sizeof(header));
    std::memcpy(dmlc::BeginPtr(*blob), &header, sizeof(header));
  }
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_IMAGE_RECORDIO_H_
