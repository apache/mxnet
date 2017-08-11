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

import java.util.List;

import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeTransformer;
import edu.stanford.nlp.util.Generics;

/**
 * This transformer collapses chains of unary nodes so that the top
 * node is the only node left.  The Sentiment model does not handle
 * unary nodes, so this simplifies them to make a binary tree consist
 * entirely of binary nodes and preterminals.  A new tree with new
 * nodes and labels is returned; the original tree is unchanged.
 *
 * @author John Bauer
 */
public class CollapseUnaryTransformer implements TreeTransformer {
  public Tree transformTree(Tree tree) {
    if (tree.isPreTerminal() || tree.isLeaf()) {
      return tree.deepCopy();
    }

    Label label = tree.label().labelFactory().newLabel(tree.label());
    Tree[] children = tree.children();
    while (children.length == 1 && !children[0].isLeaf()) {
      children = children[0].children();
    }
    List<Tree> processedChildren = Generics.newArrayList();
    for (Tree child : children) {
      processedChildren.add(transformTree(child));
    }
    return tree.treeFactory().newTreeNode(label, processedChildren);
  }
}
