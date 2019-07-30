/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet

import org.apache.mxnet.io.NDArrayIter
import org.scalatest.FunSuite
import org.slf4j.LoggerFactory

class SparseNDArraySuite  extends FunSuite {

  private val logger = LoggerFactory.getLogger(classOf[SparseNDArraySuite])

  test("create CSR NDArray") {
    val data = Array(7f, 8f, 9f)
    val indices = Array(0f, 2f, 1f)
    val indptr = Array(0f, 2f, 2f, 3f)
    val shape = Shape(3, 4)
    val sparseND = SparseNDArray.csrMatrix(data, indices, indptr, shape, Context.cpu())
    assert(sparseND.shape == Shape(3, 4))
    assert(sparseND.toArray
      sameElements Array(7.0f, 0.0f, 8.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 9.0f, 0.0f, 0.0f))
    assert(sparseND.sparseFormat == SparseFormat.CSR)
    assert(sparseND.getIndptr.toArray sameElements indptr)
    assert(sparseND.getIndices.toArray sameElements indices)
  }

  test("create Row Sparse NDArray") {
    val data = Array(
      Array(1f, 2f),
      Array(3f, 4f)
    )
    val indices = Array(1f, 4f)
    val shape = Shape(6, 2)
    val sparseND = SparseNDArray.rowSparseArray(data, indices, shape, Context.cpu())
    assert(sparseND.sparseFormat == SparseFormat.ROW_SPARSE)
    assert(sparseND.shape == Shape(6, 2))
    assert(sparseND.at(1).toArray sameElements Array(1f, 2f))
    assert(sparseND.getIndices.toArray sameElements indices)
  }

  test("Test retain") {
    val arr = Array(
      Array(1f, 2f),
      Array(3f, 4f),
      Array(5f, 6f)
    )
    val indices = Array(0f, 1f, 3f)
    val rspIn = SparseNDArray.rowSparseArray(arr, indices, Shape(4, 2), Context.cpu())
    val toRetain = Array(0f, 3f)
    val rspOut = SparseNDArray.retain(rspIn, toRetain)
    assert(rspOut.getData.toArray sameElements Array(1f, 2f, 5f, 6f))
    assert(rspOut.getIndices.toArray sameElements Array(0f, 3f))
  }

  test("Test add") {
    val nd = NDArray.array(Array(1f, 2f, 3f), Shape(3)).toSparse(Some(SparseFormat.ROW_SPARSE))
    val nd2 = nd + nd
    assert(nd2.isInstanceOf[SparseNDArray])
    assert(nd2.toArray sameElements Array(2f, 4f, 6f))
  }

  test("Test DataIter") {
    val nd = NDArray.array(Array(1f, 2f, 3f), Shape(1, 3)).toSparse(Some(SparseFormat.CSR))
    val arr = IndexedSeq(nd, nd, nd, nd)
    val iter = new NDArrayIter(arr)
    while (iter.hasNext) {
      val tempArr = iter.next().data
      tempArr.foreach(ele => {
        assert(ele.sparseFormat == SparseFormat.CSR)
        assert(ele.shape == Shape(1, 3))
      })
    }
  }


}
