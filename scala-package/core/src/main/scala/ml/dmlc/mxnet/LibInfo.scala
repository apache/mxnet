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
  @native def mxKVStoreCreate(name: String, handle: KVStoreHandle): Int
  @native def mxKVStoreInit(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            // values ought to be Array[NDArrayHandle],
                            // we pass ptr address directly for performance consideration
                            values: Array[CPtrAddress]): Int
  @native def mxKVStorePush(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            // values ought to be Array[NDArrayHandle],
                            // we pass ptr address directly for performance consideration
                            values: Array[CPtrAddress],
                            priority: Int): Int
  @native def mxKVStorePull(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            // outs ought to be Array[NDArrayHandle],
                            // we pass ptr address directly for performance consideration
                            outs: Array[CPtrAddress],
                            priority: Int): Int
  @native def mxKVStoreSetUpdater(handle: KVStoreHandle,
                                  updaterFunc: MXKVStoreUpdater,
                                  updaterHandle: AnyRef): Int
  @native def mxKVStoreIsWorkerNode(isWorker: RefInt): Int
  @native def mxKVStoreGetType(handle: KVStoreHandle, kvType: RefString): Int
  @native def mxKVStoreSendCommmandToServers(handle: KVStoreHandle,
                                             head: Int, body: String): Int
  @native def mxKVStoreBarrier(handle: KVStoreHandle): Int
  @native def mxKVStoreGetGroupSize(handle: KVStoreHandle, size: RefInt): Int
  @native def mxKVStoreGetRank(handle: KVStoreHandle, size: RefInt): Int
  //DataIter Funcs
  @native def mxListDataIters(handles: Array[DataIterCreator]): Int
  @native def mxDateIterCreateIter(handle: DataIterCreator,
                                   keys: Array[String],
                                   vals: Array[String],
                                   out: DataIterHandle): Int
  @native def mxDataIterGetIterInfo(creator: DataIterCreator,
                                    name: RefString,
                                    description: RefString,
                                    argNames: ListBuffer[String],
                                    argTypeInfos: ListBuffer[String],
                                    argDescriptions: ListBuffer[String]): Int
  @native def mxDataIterFree(handle: DataIterHandle): Int
  @native def mxDataIterBeforeFirst(handle: DataIterHandle): Int
  @native def mxDataIterNext(handle: DataIterHandle, out: RefInt): Int
  @native def mxDataIterGetLabel(handle: DataIterHandle,
                                 out: NDArrayHandle): Int
  @native def mxDataIterGetData(handle: DataIterHandle,
                                out: NDArrayHandle): Int
  @native def mxDataIterGetIndex(handle: DataIterHandle,
                                outIndex: ListBuffer[Long],
                                outSize: RefLong): Int
  @native def mxDataIterGetPadNum(handle: DataIterHandle,
                                  out: MXUintRef): Int
}
