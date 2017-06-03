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

package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
 * JNI functions
 */
private[mxnet] class LibInfo {
  @native def nativeLibInit(): Int
  @native def mxGetLastError(): String
  // Operators
  @native def mxListAllOpNames(names: ListBuffer[String]): Int
  @native def nnGetOpHandle(opName: String, opHandle: RefLong): Int
  // NDArray
  @native def mxImperativeInvoke(creator: FunctionHandle,
                                 inputs: Array[NDArrayHandle],
                                 outputsGiven: Array[NDArrayHandle],
                                 outputs: ArrayBuffer[NDArrayHandle],
                                 numParams: Int,
                                 paramKeys: Array[String],
                                 paramVals: Array[String]): Int
  @native def mxNDArrayFree(handle: NDArrayHandle): Int
  @native def mxNDArrayCreateNone(out: NDArrayHandleRef): Int
  @native def mxNDArrayCreateEx(shape: Array[Int],
                                ndim: Int,
                                devType: Int,
                                devId: Int,
                                delayAlloc: Int,
                                dtype: Int,
                                out: NDArrayHandleRef): Int
  @native def mxNDArrayWaitAll(): Int
  @native def mxNDArrayWaitToRead(handle: NDArrayHandle): Int
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
                           useVars: Array[NDArrayHandle],
                           scalarArgs: Array[MXFloat],
                           mutateVars: Array[NDArrayHandle]): Int
  @native def mxFuncInvokeEx(function: FunctionHandle,
                             useVars: Array[NDArrayHandle],
                             scalarArgs: Array[MXFloat],
                             mutateVars: Array[NDArrayHandle],
                             numParams: Int,
                             paramKeys: Array[Array[Byte]],
                             paramVals: Array[Array[Byte]]): Int
  @native def mxNDArrayGetShape(handle: NDArrayHandle,
                                ndim: MXUintRef,
                                data: ArrayBuffer[Int]): Int
  @native def mxNDArraySyncCopyToCPU(handle: NDArrayHandle,
                                     data: Array[Byte],
                                     size: Int): Int
  @native def mxNDArraySlice(handle: NDArrayHandle,
                             start: MXUint,
                             end: MXUint,
                             sliceHandle: NDArrayHandleRef): Int
  @native def mxNDArrayAt(handle: NDArrayHandle,
                          idx: MXUint,
                          out: NDArrayHandleRef): Int
  @native def mxNDArrayReshape(handle: NDArrayHandle,
                               nDim: Int,
                               dims: Array[Int],
                               reshapeHandle: NDArrayHandleRef): Int
  @native def mxNDArraySyncCopyFromCPU(handle: NDArrayHandle,
                                       source: Array[MXFloat],
                                       size: Int): Int
  @native def mxNDArrayLoad(fname: String,
                            outSize: MXUintRef,
                            handles: ArrayBuffer[NDArrayHandle],
                            outNameSize: MXUintRef,
                            names: ArrayBuffer[String]): Int
  @native def mxNDArraySave(fname: String,
                            handles: Array[NDArrayHandle],
                            keys: Array[String]): Int
  @native def mxNDArrayGetContext(handle: NDArrayHandle, devTypeId: RefInt, devId: RefInt): Int
  @native def mxNDArraySaveRawBytes(handle: NDArrayHandle, buf: ArrayBuffer[Byte]): Int
  @native def mxNDArrayLoadFromRawBytes(bytes: Array[Byte], handle: NDArrayHandleRef): Int
  @native def mxNDArrayGetDType(handle: NDArrayHandle, dtype: RefInt): Int

  // KVStore Server
  @native def mxInitPSEnv(keys: Array[String], values: Array[String]): Int
  @native def mxKVStoreRunServer(handle: KVStoreHandle, controller: KVServerControllerCallback): Int
  @native def mxKVStoreGetNumDeadNode(handle: KVStoreHandle, nodeId: Int, number: RefInt): Int

  // KVStore
  @native def mxKVStoreCreate(name: String, handle: KVStoreHandleRef): Int
  @native def mxKVStoreInit(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            values: Array[NDArrayHandle]): Int
  @native def mxKVStorePush(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            values: Array[NDArrayHandle],
                            priority: Int): Int
  @native def mxKVStorePull(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            outs: Array[NDArrayHandle],
                            priority: Int): Int
  @native def mxKVStoreSetUpdater(handle: KVStoreHandle, updaterFunc: MXKVStoreUpdater): Int
  @native def mxKVStoreIsWorkerNode(isWorker: RefInt): Int
  @native def mxKVStoreGetType(handle: KVStoreHandle, kvType: RefString): Int
  @native def mxKVStoreSendCommmandToServers(handle: KVStoreHandle,
                                             head: Int, body: String): Int
  @native def mxKVStoreBarrier(handle: KVStoreHandle): Int
  @native def mxKVStoreGetGroupSize(handle: KVStoreHandle, size: RefInt): Int
  @native def mxKVStoreGetRank(handle: KVStoreHandle, size: RefInt): Int
  @native def mxKVStoreSetBarrierBeforeExit(handle: KVStoreHandle, doBarrier: Int): Int
  @native def mxKVStoreFree(handle: KVStoreHandle): Int

  // DataIter Funcs
  @native def mxListDataIters(handles: ListBuffer[DataIterCreator]): Int
  @native def mxDataIterCreateIter(handle: DataIterCreator,
                                   keys: Array[String],
                                   vals: Array[String],
                                   out: DataIterHandleRef): Int
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
                                 out: NDArrayHandleRef): Int
  @native def mxDataIterGetData(handle: DataIterHandle,
                                out: NDArrayHandleRef): Int
  @native def mxDataIterGetIndex(handle: DataIterHandle,
                                outIndex: ListBuffer[Long],
                                outSize: RefLong): Int
  @native def mxDataIterGetPadNum(handle: DataIterHandle,
                                  out: MXUintRef): Int
  // Executors
  @native def mxExecutorOutputs(handle: ExecutorHandle, outputs: ArrayBuffer[NDArrayHandle]): Int
  @native def mxExecutorFree(handle: ExecutorHandle): Int
  @native def mxExecutorForward(handle: ExecutorHandle, isTrain: Int): Int
  @native def mxExecutorBackward(handle: ExecutorHandle,
                                 grads: Array[NDArrayHandle]): Int
  @native def mxExecutorPrint(handle: ExecutorHandle, debugStr: RefString): Int
  @native def mxExecutorSetMonitorCallback(handle: ExecutorHandle, callback: MXMonitorCallback): Int

  // Symbols
  @native def mxSymbolListAtomicSymbolCreators(symbolList: ListBuffer[SymbolHandle]): Int
  @native def mxSymbolGetAtomicSymbolInfo(handle: SymbolHandle,
                                          name: RefString,
                                          desc: RefString,
                                          numArgs: MXUintRef,
                                          argNames: ListBuffer[String],
                                          argTypes: ListBuffer[String],
                                          argDescs: ListBuffer[String],
                                          keyVarNumArgs: RefString): Int
  @native def mxSymbolCreateAtomicSymbol(handle: SymbolHandle,
                                         paramKeys: Array[String],
                                         paramVals: Array[String],
                                         symHandleRef: SymbolHandleRef): Int
  @native def mxSymbolSetAttr(handle: SymbolHandle, key: String, value: String): Int
  @native def mxSymbolListAttrShallow(handle: SymbolHandle,
                                      outSize: MXUintRef,
                                      out: ArrayBuffer[String]): Int
  @native def mxSymbolListAttr(handle: SymbolHandle,
                               outSize: MXUintRef,
                               out: ArrayBuffer[String]): Int
  @native def mxSymbolCompose(handle: SymbolHandle,
                              name: String,
                              keys: Array[String],
                              args: Array[SymbolHandle]): Int
  @native def mxSymbolCreateVariable(name: String, out: SymbolHandleRef): Int
  @native def mxSymbolGetAttr(handle: SymbolHandle,
                              key: String,
                              ret: RefString,
                              success: RefInt): Int
  @native def mxSymbolListArguments(handle: SymbolHandle,
                                    arguments: ArrayBuffer[String]): Int
  @native def mxSymbolCopy(handle: SymbolHandle, clonedHandle: SymbolHandleRef): Int
  @native def mxSymbolListAuxiliaryStates(handle: SymbolHandle,
                                          arguments: ArrayBuffer[String]): Int
  @native def mxSymbolListOutputs(handle: SymbolHandle,
                                  outputs: ArrayBuffer[String]): Int
  @native def mxSymbolCreateGroup(handles: Array[SymbolHandle], out: SymbolHandleRef): Int
  @native def mxSymbolPrint(handle: SymbolHandle, str: RefString): Int
  @native def mxSymbolGetInternals(handle: SymbolHandle, out: SymbolHandleRef): Int
  @native def mxSymbolInferType(handle: SymbolHandle,
                                keys: Array[String],
                                sdata: Array[Int],
                                argTypeData: ListBuffer[Int],
                                outTypeData: ListBuffer[Int],
                                auxTypeData: ListBuffer[Int],
                                complete: RefInt): Int
  @native def mxSymbolInferShape(handle: SymbolHandle,
                                 numArgs: MXUint,
                                 keys: Array[String],
                                 argIndPtr: Array[MXUint],
                                 argShapeData: Array[MXUint],
                                 inShapeData: ListBuffer[Array[Int]],
                                 outShapeData: ListBuffer[Array[Int]],
                                 auxShapeData: ListBuffer[Array[Int]],
                                 complete: RefInt): Int
  @native def mxSymbolGetOutput(handle: SymbolHandle, index: Int, out: SymbolHandleRef): Int
  @native def mxSymbolSaveToJSON(handle: SymbolHandle, out: RefString): Int
  @native def mxSymbolCreateFromJSON(json: String, handle: SymbolHandleRef): Int
  // scalastyle:off parameterNum
  @native def mxExecutorBindX(handle: SymbolHandle,
                              deviceTypeId: Int,
                              deviceID: Int,
                              numCtx: Int,
                              ctxMapKeys: Array[String],
                              ctxMapDevTypes: Array[Int],
                              ctxMapDevIDs: Array[Int],
                              numArgs: Int,
                              argsHandle: Array[NDArrayHandle],
                              argsGradHandle: Array[NDArrayHandle],
                              reqsArray: Array[Int],
                              auxArgsHandle: Array[NDArrayHandle],
                              out: ExecutorHandleRef): Int
  @native def mxExecutorBindEX(handle: SymbolHandle,
                              deviceTypeId: Int,
                              deviceID: Int,
                              numCtx: Int,
                              ctxMapKeys: Array[String],
                              ctxMapDevTypes: Array[Int],
                              ctxMapDevIDs: Array[Int],
                              numArgs: Int,
                              argsHandle: Array[NDArrayHandle],
                              argsGradHandle: Array[NDArrayHandle],
                              reqsArray: Array[Int],
                              auxArgsHandle: Array[NDArrayHandle],
                              sharedExec: ExecutorHandle,
                              out: ExecutorHandleRef): Int
  // scalastyle:on parameterNum
  @native def mxSymbolSaveToFile(handle: SymbolHandle, fname: String): Int
  @native def mxSymbolCreateFromFile(fname: String, handle: SymbolHandleRef): Int
  @native def mxSymbolFree(handle: SymbolHandle): Int

  // Random
  @native def mxRandomSeed(seed: Int): Int

  @native def mxNotifyShutdown(): Int

  // RecordIO
  @native def mxRecordIOWriterCreate(uri: String, out: RecordIOHandleRef): Int
  @native def mxRecordIOReaderCreate(uri: String, out: RecordIOHandleRef): Int
  @native def mxRecordIOWriterFree(handle: RecordIOHandle): Int
  @native def mxRecordIOReaderFree(handle: RecordIOHandle): Int
  @native def mxRecordIOWriterWriteRecord(handle: RecordIOHandle, buf: String, size: Int): Int
  @native def mxRecordIOReaderReadRecord(handle: RecordIOHandle, buf: RefString): Int
  @native def mxRecordIOWriterTell(handle: RecordIOHandle, pos: RefInt): Int
  @native def mxRecordIOReaderSeek(handle: RecordIOHandle, pos: Int): Int

  // Rtc
  @native def mxRtcCreate(name: String,
                          inputNames: Array[String],
                          outputNames: Array[String],
                          inputs: Array[NDArrayHandle],
                          outputs: Array[NDArrayHandle],
                          kernel: String,
                          out: RtcHandleRef): Int
  @native def mxRtcPush(handle: RtcHandle,
                        inputs: Array[NDArrayHandle],
                        outputs: Array[NDArrayHandle],
                        gridDimX: Int,
                        gridDimY: Int,
                        gridDimZ: Int,
                        blockDimX: Int,
                        blockDimY: Int,
                        blockDimZ: Int): Int
  @native def mxRtcFree(handle: RtcHandle): Int

  // CustomOp
  @native def mxCustomOpRegister(regName: String, opProp: CustomOpProp): Int

  // Profiler
  @native def mxSetProfilerConfig(mode: Int, fileName: String): Int
  @native def mxSetProfilerState(state: Int): Int
  @native def mxDumpProfile(): Int
}
