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
 * \file bounding_box-common.h
 * \brief bounding box util functions and operators commonly used by CPU and GPU implementations
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_COMMON_H_
#define MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_COMMON_H_
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace box_common_enum {
enum BoxType {kCorner, kCenter};
}

// compute line intersect along either height or width
template<typename DType>
MSHADOW_XINLINE DType Intersect(const DType *a, const DType *b, int encode) {
  DType a1 = a[0];
  DType a2 = a[2];
  DType b1 = b[0];
  DType b2 = b[2];
  DType w;
  if (box_common_enum::kCorner == encode) {
    DType left = a1 > b1 ? a1 : b1;
    DType right = a2 < b2 ? a2 : b2;
    w = right - left;
  } else {
    DType aw = a2 / 2;
    DType bw = b2 / 2;
    DType al = a1 - aw;
    DType ar = a1 + aw;
    DType bl = b1 - bw;
    DType br = b1 + bw;
    DType left = bl > al ? bl : al;
    DType right = br < ar ? br : ar;
    w = right - left;
  }
  return w > 0 ? w : DType(0);
}

/*!
   * \brief Implementation of the non-maximum suppression operation
   *
   * \param i the launched thread index
   * \param index sorted index in descending order
   * \param batch_start map (b, k) to compact index by indices[batch_start[b] + k]
   * \param input the input of nms op
   * \param areas pre-computed box areas
   * \param k nms topk number
   * \param ref compare reference position
   * \param num number of input boxes in each batch
   * \param stride input stride, usually 6 (id-score-x1-y1-x2-y2)
   * \param offset_box box offset, usually 2
   * \param thresh nms threshold
   * \param force force suppress regardless of class id
   * \param offset_id class id offset, used when force == false, usually 0
   * \param encode box encoding type, corner(0) or center(1)
   * \param DType the data type
   */
struct nms_impl {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int32_t *index, const int32_t *batch_start,
                                  const DType *input, const DType *areas,
                                  int k, int ref, int num,
                                  int stride, int offset_box, int offset_id,
                                  float thresh, bool force, int encode) {
    int b = i / k;  // batch
    int pos = i % k + ref + 1;  // position
    ref = static_cast<int>(batch_start[b]) + ref;
    pos = static_cast<int>(batch_start[b]) + pos;
    if (ref >= static_cast<int>(batch_start[b + 1])) return;
    if (pos >= static_cast<int>(batch_start[b + 1])) return;
    if (index[ref] < 0) return;  // reference has been suppressed
    if (index[pos] < 0) return;  // self been suppressed
    int ref_offset = static_cast<int>(index[ref]) * stride + offset_box;
    int pos_offset = static_cast<int>(index[pos]) * stride + offset_box;
    if (!force && offset_id >=0) {
      int ref_id = static_cast<int>(input[ref_offset - offset_box + offset_id]);
      int pos_id = static_cast<int>(input[pos_offset - offset_box + offset_id]);
      if (ref_id != pos_id) return;  // different class
    }
    DType intersect = Intersect(input + ref_offset, input + pos_offset, encode);
    intersect *= Intersect(input + ref_offset + 1, input + pos_offset + 1, encode);
    int ref_area_offset = static_cast<int>(index[ref]);
    int pos_area_offset = static_cast<int>(index[pos]);
    DType iou = intersect / (areas[ref_area_offset] + areas[pos_area_offset] - intersect);
    if (iou > thresh) {
      index[pos] = -1;
    }
  }
};

namespace mshadow_op {
struct less_than : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return static_cast<DType>(a < b);
  }
};

struct greater_than : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return static_cast<DType>(a > b);
  }
};

struct not_equal : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return static_cast<DType>(a != b);
  }
};

struct bool_and : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return static_cast<DType>(a && b);
  }
};
}   // namespace mshadow_op

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_COMMON_H_
