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
 * \file index_copy-inl.h
 * \brief implementation of neighbor_sample tensor operation
 */

#ifndef MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_
#define MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "../operator_common.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <queue>

namespace mxnet {
namespace op {

typedef int64_t dgl_id_t;

// Input
//------------------------------------------------------------------------------
// input[0]: Graph
// input[1]: seed_vertices
// args[0]: num_hops
// args[1]: num_neighbor 
// args[2]: max_num_vertices
//------------------------------------------------------------------------------

// Output
//------------------------------------------------------------------------------
// output[0]: sampled_vertices
// output[1]: sampled_csr_graph
//------------------------------------------------------------------------------

// For BFS traversal
struct ver_node {
  dgl_id_t vertex_id;
  int level;
}; 

// How to set the default value?
struct NeighborSampleParam : public dmlc::Parameter<NeighborSampleParam> {
  dgl_id_t num_hops, num_neighbor, max_num_vertices;
  DMLC_DECLARE_PARAMETER(NeighborSampleParam) {
    DMLC_DECLARE_FIELD(num_hops)
      .set_default(1)
      .describe("Number of hops.");
    DMLC_DECLARE_FIELD(num_neighbor)
      .set_default(2)
      .describe("Number of neighbor.");
    DMLC_DECLARE_FIELD(max_num_vertices)
      .set_default(100)
      .describe("Max number of vertices.");
  }
};

static bool CSRNeighborSampleStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 2);

  CHECK_EQ(in_attrs->at(0), mxnet::kCSRStorage);
  CHECK_EQ(in_attrs->at(1), mxnet::kDefaultStorage);

  bool success = true;
  if (!type_assign(&(*out_attrs)[0], mxnet::kDefaultStorage)) {
    success = false;
  }
  if (!type_assign(&(*out_attrs)[1], mxnet::kCSRStorage)) {
    success = false;
  }

  *dispatch_mode = DispatchMode::kFComputeEx;

  return success;
}

static bool CSRNeighborSampleShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape> *in_attrs,
                                   std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 2);

  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);
  // Check the graph shape
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(0)[1]);

  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  TShape out_shape(1);
  out_shape[0] = params.max_num_vertices;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);

  TShape out_csr_shape(2);
  out_csr_shape[0] = params.max_num_vertices;
  out_csr_shape[1] = in_attrs->at(0)[1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_csr_shape);

  return out_attrs->at(0).ndim() != 0U &&
         out_attrs->at(0).Size() != 0U &&
         out_attrs->at(1).ndim() != 0U &&
         out_attrs->at(1).Size() != 0U;
}

static bool CSRNeighborSampleType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 2);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));

  return out_attrs->at(0) != -1;
}

static void GetSrcList(const dgl_id_t* val_list,
                       const dgl_id_t* col_list,
                       const dgl_id_t* indptr,
                       const dgl_id_t dst_id,
                       std::vector<dgl_id_t>& src_list,
                       std::vector<dgl_id_t>& edge_list) {
  for (dgl_id_t i = *(indptr+dst_id); i < *(indptr+dst_id+1); ++i) {
    src_list.push_back(col_list[i]);
    edge_list.push_back(val_list[i]);
  }
}

static void GetSample(std::vector<dgl_id_t>& ver_list,
                      std::vector<dgl_id_t>& edge_list,
                      const size_t max_num_neighbor,
                      std::vector<dgl_id_t>& out,
                      std::vector<dgl_id_t>& out_edge) {
  CHECK_EQ(ver_list.size(), edge_list.size());
  // Copy ver_list to output
  if (ver_list.size() <= max_num_neighbor) {
    for (size_t i = 0; i < ver_list.size(); ++i) {
      out.push_back(ver_list[i]);
      out_edge.push_back(edge_list[i]);
    }
    return;
  }
  // Make sample
  std::unordered_map<size_t, bool> mp;
  size_t sample_count = 0;
  for (;;) {
    // rand_num = [0, ver_list.size()-1]
    size_t rand_num = rand() % ver_list.size(); 
    auto got = mp.find(rand_num);
    if (got != mp.end() && mp[rand_num]) {
      // re-sample
      continue;
    }
    mp[rand_num] = true;
    out.push_back(ver_list[rand_num]);
    out_edge.push_back(edge_list[rand_num]);
    sample_count++;
    if (sample_count == max_num_neighbor) {
      break;
    }
  }
}

static void CSRNeighborSampleComputeExCPU(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);

  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  // set seed for random sampling
  srand(time(nullptr));

  dgl_id_t num_hops = params.num_hops;
  dgl_id_t num_neighbor = params.num_neighbor;
  dgl_id_t max_num_vertices = params.max_num_vertices;

  size_t seed_num = inputs[1].data().Size();

  CHECK_GE(max_num_vertices, seed_num);

  const dgl_id_t* val_list = inputs[0].data().dptr<dgl_id_t>();
  const dgl_id_t* col_list = inputs[0].aux_data(csr::kIdx).dptr<dgl_id_t>();
  const dgl_id_t* indptr = inputs[0].aux_data(csr::kIndPtr).dptr<dgl_id_t>();
  const dgl_id_t* seed = inputs[1].data().dptr<dgl_id_t>();

  dgl_id_t* out = outputs[0].data().dptr<dgl_id_t>();

  // BFS traverse the graph and sample vertices
  dgl_id_t sub_vertices_count = 0;
  std::map<dgl_id_t, bool> sub_ver_mp;
  std::queue<ver_node> node_queue;
  // add seed vertices
  for (size_t i = 0; i < seed_num; ++i) {
    ver_node node;
    node.vertex_id = seed[i];
    node.level = 0;
    node_queue.push(node);
    sub_ver_mp[node.vertex_id] = true;
    sub_vertices_count++;
  }

  std::vector<dgl_id_t> tmp_src_list;
  std::vector<dgl_id_t> tmp_edge_list;
  std::vector<dgl_id_t> tmp_sampled_src_list;
  std::vector<dgl_id_t> tmp_sampled_edge_list;

  std::map<dgl_id_t, std::vector<dgl_id_t> > ver_mp;
  std::map<dgl_id_t, std::vector<dgl_id_t> > edge_mp;

  while (!node_queue.empty()) {
    ver_node& cur_node = node_queue.front();
    if (cur_node.level < num_hops) {

      dgl_id_t dst_id = cur_node.vertex_id;
      tmp_src_list.clear();
      tmp_edge_list.clear();
      tmp_sampled_src_list.clear();
      tmp_sampled_edge_list.clear();

      GetSrcList(val_list, 
                 col_list, 
                 indptr, 
                 dst_id, 
                 tmp_src_list, 
                 tmp_edge_list);

      GetSample(tmp_src_list, 
                tmp_edge_list, 
                num_neighbor, 
                tmp_sampled_src_list,
                tmp_sampled_edge_list);

      ver_mp[dst_id] = tmp_sampled_src_list;
      edge_mp[dst_id] = tmp_sampled_edge_list;
      
      sub_vertices_count++;
      if (sub_vertices_count == max_num_vertices) {
        break;
      }

      for (size_t i = 0; i < tmp_sampled_src_list.size(); ++i) {
        auto got = sub_ver_mp.find(tmp_sampled_src_list[i]);
        if (got == sub_ver_mp.end()) {
          sub_ver_mp[tmp_sampled_src_list[i]] = true;
          sub_vertices_count++;
          ver_node new_node;
          new_node.vertex_id = tmp_sampled_src_list[i];
          new_node.level = cur_node.level + 1;
          node_queue.push(new_node);
        }
      }
    }
    node_queue.pop();
  }

  // Copy sub_ver_mp to output[0]
  dgl_id_t idx = 0;
  for (auto& data: sub_ver_mp) {
    if (data.second) {
      *(out+idx) = data.first;
      idx++;
    }
  }
  // The rest data will be set to -1
  for (dgl_id_t i = idx; i < max_num_vertices; ++i) {
    *(out+i) = -1;
  }

  // Construct sub_csr_graph
  std::vector<dgl_id_t> sub_val;
  std::vector<dgl_id_t> sub_col_list;
  std::vector<dgl_id_t> sub_indptr(max_num_vertices+1, 0);

  size_t index = 1;
  for (auto& data: sub_ver_mp) {
    dgl_id_t dst_id = data.first;
    auto edge = edge_mp.find(dst_id);
    auto vert = ver_mp.find(dst_id);
    if (edge != edge_mp.end() && vert != ver_mp.end()) {
      CHECK_EQ(edge->second.size(), vert->second.size());
      for (auto& val : edge->second) {
        sub_val.push_back(val);
      }
      for (auto& val : vert->second) {
        sub_col_list.push_back(val);
      }
      sub_indptr[index] = sub_indptr[index-1] + edge->second.size();
    } else {
      sub_indptr[index] = sub_indptr[index-1];
    }
    index++;
  }

  // Copy sub_csr_graph to output[1]
  const NDArray& sub_csr = outputs[1];
  TShape shape_1(1);
  TShape shape_2(1);
  shape_1[0] = sub_val.size();
  shape_2[0] = sub_indptr.size();
  sub_csr.CheckAndAllocData(shape_1);
  sub_csr.CheckAndAllocAuxData(csr::kIdx, shape_1);
  sub_csr.CheckAndAllocAuxData(csr::kIndPtr, shape_2);

  dgl_id_t* val_list_out = sub_csr.data().dptr<dgl_id_t>();
  dgl_id_t* col_list_out = sub_csr.aux_data(1).dptr<dgl_id_t>();
  dgl_id_t* indptr_out = sub_csr.aux_data(0).dptr<dgl_id_t>();


  std::copy(sub_val.begin(), sub_val.end(), val_list_out);
  std::copy(sub_col_list.begin(), sub_col_list.end(), col_list_out);
  std::copy(sub_indptr.begin(), sub_indptr.end(), indptr_out);
}

}  // op
}  // mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_
