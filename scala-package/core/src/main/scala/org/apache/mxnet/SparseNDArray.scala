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

import org.apache.mxnet.Base.{NDArrayHandle, NDArrayHandleRef, checkCall, _LIB}
import org.apache.mxnet.DType.DType
import org.apache.mxnet.SparseFormat.SparseFormat

object SparseNDArray {
  /**
    * Create a Compressed Sparse Row Storage (CSR) Format Matrix
    * @param data the data to feed
    * @param indices The indices array stores the column index for each non-zero element in data
    * @param indptr The indptr array is what will help identify the rows where the data appears
    * @param shape the shape of CSR NDArray to be created
    * @param ctx the context of this NDArray
    * @return SparseNDArray
    */
  def csrMatrix(data: Array[Float], indices: Array[Float],
                indptr: Array[Float], shape: Shape, ctx: Context): SparseNDArray = {
    val fmt = SparseFormat.CSR
    val dataND = NDArray.array(data, Shape(data.length), ctx)
    val indicesND = NDArray.array(indices, Shape(indices.length), ctx).asType(DType.Int64)
    val indptrND = NDArray.array(indptr, Shape(indptr.length), ctx).asType(DType.Int64)
    val dTypes = Array(indptrND.dtype, indicesND.dtype)
    val shapes = Array(indptrND.shape, indicesND.shape)
    val handle =
      newAllocHandle(fmt, shape, ctx, false, DType.Float32, dTypes, shapes)
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, dataND.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indptrND.handle, 0))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indicesND.handle, 1))
    new SparseNDArray(handle)
  }

  /**
    * RowSparseNDArray stores the matrix in row sparse format,
    * which is designed for arrays of which most row slices are all zeros
    * @param data Any Array(Array(... Array(Float)))
    * @param indices the indices to store the data
    * @param shape shape of the NDArray
    * @param ctx Context
    * @return SparseNDArray
    */
  def rowSparseArray(data: Array[_], indices: Array[Float],
                     shape: Shape, ctx: Context): SparseNDArray = {
    val dataND = NDArray.toNDArray(data)
    val indicesND = NDArray.array(indices, Shape(indices.length), ctx).asType(DType.Int64)
    rowSparseArray(dataND, indicesND, shape, ctx)
  }

  /**
    * RowSparseNDArray stores the matrix in row sparse format,
    * which is designed for arrays of which most row slices are all zeros
    * @param data NDArray input
    * @param indices in NDArray. Only DType.Int64 supported
    * @param shape shape of the NDArray
    * @param ctx Context
    * @return
    */
  def rowSparseArray(data: NDArray, indices: NDArray,
                     shape: Shape, ctx: Context): SparseNDArray = {
    val fmt = SparseFormat.ROW_SPARSE
    val handle = newAllocHandle(fmt, shape, ctx, false,
      DType.Float32, Array(indices.dtype), Array(indices.shape))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, data.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indices.handle, 0))
    new SparseNDArray(handle)
  }

  def retain(sparseNDArray: SparseNDArray, indices: Array[Float]): SparseNDArray = {
    if (sparseNDArray.sparseFormat == SparseFormat.CSR) {
      throw new IllegalArgumentException("CSR not supported")
    }
    NDArray.genericNDArrayFunctionInvoke("_sparse_retain",
      Seq(sparseNDArray, NDArray.toNDArray(indices))).head.toSparse()
  }

  private def newAllocHandle(stype : SparseFormat,
                             shape: Shape,
                             ctx: Context,
                             delayAlloc: Boolean,
                             dtype: DType = DType.Float32,
                             auxDTypes: Array[DType],
                             auxShapes: Array[Shape]) : NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateSparseEx(
      stype.id,
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      dtype.id,
      auxDTypes.length,
      auxDTypes.map(_.id),
      auxShapes.map(_.length),
      auxShapes.map(_.get(0)),
      hdl)
    )
    hdl.value
  }
}

/**
  * Sparse NDArray is the child class of NDArray designed to hold the Sparse format
  *
  * <p> Currently, Rowsparse and CSR typed NDArray is supported. Most of the Operators
  * will convert Sparse NDArray to dense. Basic operators like <code>add</code> will
  * have optimization for sparse operattions</p>
  * @param handle The pointer that SparseNDArray holds
  * @param writable whether the NDArray is writable
  */
class SparseNDArray private[mxnet] (override private[mxnet] val handle: NDArrayHandle,
                                    override val writable: Boolean = true)
  extends NDArray(handle, writable) {

  private lazy val dense: NDArray = toDense

  override def toString: String = {
    dense.toString
  }

  /**
    * Convert a SparseNDArray to dense NDArray
    * @return NDArray
    */
  def toDense: NDArray = {
      NDArray.api.cast_storage(this, SparseFormat.DEFAULT.toString).head
  }

  override def toArray: Array[Float] = {
    dense.toArray
  }

  override def at(idx: Int): NDArray = {
    dense.at(idx)
  }

  override def slice(start: Int, end: Int): NDArray = {
    NDArray.api.slice(this, Shape(start), Shape(end))
  }

  /**
    * Get the Data portion from a Row Sparse NDArray
    * @return NDArray
    */
  def getData: NDArray = {
    require(this.sparseFormat == SparseFormat.ROW_SPARSE, "Not Supported for CSR")
    val handle = new NDArrayHandleRef
    _LIB.mxNDArrayGetDataNDArray(this.handle, handle)
    new NDArray(handle.value, false)
  }

  /**
    * Get the indptr Array from a CSR NDArray
    * @return NDArray
    */
  def getIndptr: NDArray = {
    require(this.sparseFormat == SparseFormat.CSR, "Not Supported for row sparse")
    getAuxNDArray(0)
  }

  /**
    * Get the indice Array
    * @return NDArray
    */
  def getIndices: NDArray = {
    if (this.sparseFormat == SparseFormat.ROW_SPARSE) {
      getAuxNDArray(0)
    } else {
      getAuxNDArray(1)
    }
  }

  private def getAuxNDArray(idx: Int): NDArray = {
    val handle = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayGetAuxNDArray(this.handle, idx, handle))
    new NDArray(handle.value, false)
  }

}
