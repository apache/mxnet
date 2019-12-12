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

#ifndef MXNET_OPERATOR_SUBGRAPH_CUSTOM_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_CUSTOM_SUBGRAPH_PROPERTY_H_

namespace mxnet {
  namespace op {

    /*
     * This selects nodes for a subgraph
     */
    class CustomContainOpSelector: public SubgraphSelector {
    public:
      explicit CustomContainOpSelector() {}

      virtual bool Select(const nnvm::Node &n) {
        return false;
      }

      virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) {
        return false;
      }

      virtual bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) {
        return false;
      }
    };

    /*
     * This subgraph property finds a subgraph
     */
    class  CustomSubgraphProperty: public SubgraphProperty {
    public:
      // create custom subgraph property
      static SubgraphPropertyPtr Create() {
        return std::make_shared<CustomSubgraphProperty>();
      }

      // override CreateSubgraphNode
      virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                               const int subgraph_id = 0) const {
        nnvm::NodePtr n = nnvm::Node::Create();
        n->attrs.op = Op::Get("_op");
        n->attrs.name = "_op" + std::to_string(subgraph_id);
        n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
        return n;
      }

      // override CreateSubgraphSelector
      virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
        return std::make_shared<CustomContainOpSelector>();                                 
      }
    };

    MXNET_REGISTER_SUBGRAPH_PROPERTY(customProp, CustomSubgraphProperty);

  }  // namespace op
}  // namespace mxnet

#endif
