package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

// JNI functions
class LibInfo {
  @native def mxNDArrayFree(handle: NDArrayHandle): Int
  @native def mxGetLastError(): String
  @native def mxNDArrayCreateNone(out: NDArrayHandle): Int
  @native def mxNDArrayCreate(shape: Array[Int],
                              ndim: Int,
                              devType: Int,
                              devId: Int,
                              delayAlloc: Int,
                              out: NDArrayHandle): Int
  @native def mxNDArrayWaitAll(): Int
  @native def mxListFunctions(functions: ListBuffer[FunctionHandle]): Int
  @native def mxFuncDescribe(handle: FunctionHandle,
                             nUsedVars: MXUintRef,
                             nScalars: MXUintRef,
                             nMutateVars: MXUintRef,
                             typeMask: RefInt): Int
  @native def mxFuncGetInfo(handle: FunctionHandle,
                            name: RefString,
                            desc: RefString,
                            numArgs: MXUintRef,
                            argNames: ListBuffer[String],
                            argTypes: ListBuffer[String],
                            argDescs: ListBuffer[String]): Int
  @native def mxFuncInvoke(function: FunctionHandle,
                           // useVars ought to be Array[NDArrayHandle],
                           // we pass ptr address directly for performance consideration
                           useVars: Array[CPtrAddress],
                           scalarArgs: Array[MXFloat],
                           // mutateVars ought to be Array[NDArrayHandle],
                           // we pass ptr address directly for performance consideration
                           mutateVars: Array[CPtrAddress]): Int
  @native def mxNDArrayGetShape(handle: NDArrayHandle,
                                ndim: MXUintRef,
                                data: ArrayBuffer[Int]): Int
  @native def mxNDArraySyncCopyToCPU(handle: NDArrayHandle,
                                     data: Array[Float],
                                     size: Int): Int
  @native def mxNDArraySlice(handle: NDArrayHandle,
                             start: MXUint,
                             end: MXUint,
                             sliceHandle: NDArrayHandle): Int
}
