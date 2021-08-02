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

package org.apache.mxnet.jna;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.DeviceType;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.exception.JnaCallException;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.nn.SymbolBlock;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class JnaUtils {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtils.class);

    public static final MxnetLibrary LIB = LibUtils.loadLibrary();

    public static final ObjectPool<PointerByReference> REFS =
            new ObjectPool<>(PointerByReference::new, r -> r.setValue(null));

    private static final String[] OP_NAME_PREFIX = {
        "_contrib_", "_linalg_", "_sparse_", "_image_", "_random_"
    };

    private static final Map<String, FunctionInfo> OPS = getNdArrayFunctions();
    //    private static final Map<String, FunctionInfo> OPS = null;

    private static final Set<String> FEATURES = getFeaturesInternal();

    public static final String[] EMPTY_ARRAY = new String[0];
    // TODO
    /** An enum that enumerates the statuses of numpy mode. */
    public enum NumpyMode {
        OFF,
        THREAD_LOCAL_ON,
        GLOBAL_ON
    }

    public static void waitAll() {
        checkCall(LIB.MXNDArrayWaitAll());
    }

    public static void init() {
        Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD
    }

    public static void setNumpyMode(NumpyMode mode) {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(LIB.MXSetIsNumpyShape(mode.ordinal(), ret));
    }

    /**
     * ***************************************************************************** About CacheOp
     * ****************************************************************************
     */
    // TODO
    public static CachedOp createCachedOp(SymbolBlock block, MxResource parent) {
        Symbol symbol = block.getSymbol();

        List<Parameter> parameters = block.getAllParameters();

        // record data index in all inputs
        PairList<String, Integer> dataIndices = new PairList<>();
        // record parameter index in all inputs
        List<Integer> paramIndices = new ArrayList<>();
        int index = 0;
        for (Parameter parameter : parameters) {
            // We assume uninitialized parameters are data inputs
            if (parameter.isInitialized()) {
                paramIndices.add(index);
            } else {
                dataIndices.add(parameter.getName(), index);
            }
            ++index;
        }

        // Creating CachedOp
        Pointer symbolHandle = symbol.getHandle();
        PointerByReference ref = REFS.acquire();

        // static_alloc and static_shape are enabled by default
        String[] keys = {"data_indices", "param_indices", "static_alloc", "static_shape"};
        String[] values = {dataIndices.values().toString(), paramIndices.toString(), "1", "1"};

        checkCall(LIB.MXCreateCachedOp(symbolHandle, keys.length, keys, values, ref, (byte) 0));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);

        return new CachedOp(parent, pointer, parameters, paramIndices, dataIndices);
    }

    public static void freeCachedOp(Pointer handle) {
        checkCall(LIB.MXFreeCachedOp(handle));
    }

    /**
     * ***************************************************************************** About Symbol
     * ****************************************************************************
     */
    public static Pointer createSymbolFromFile(String path) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromFile(path, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer createSymbolFromString(String json) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromJSON(json, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static String[] listSymbolOutputs(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListOutputs(symbol, size, ref));
        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String printSymbol(Pointer symbol) {
        String[] outStr = new String[1];
        checkCall(LIB.NNSymbolPrint(symbol, outStr));
        return outStr[0];
    }

    public static void freeSymbol(Pointer symbol) {
        checkCall(LIB.NNSymbolFree(symbol));
    }

    public static String[] listSymbolArguments(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListArguments(symbol, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String[] listSymbolAuxiliaryStates(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListAuxiliaryStates(symbol, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String[] listSymbolNames(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.NNSymbolListInputNames(symbol, 0, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static Pointer getSymbolInternals(Pointer symbol) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolGetInternals(symbol, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    private static List<Shape> recoverShape(
            NativeSizeByReference size, PointerByReference nDim, PointerByReference data) {
        int shapeLength = (int) size.getValue().longValue();
        if (shapeLength == 0) {
            return new ArrayList<>();
        }
        int[] dims = nDim.getValue().getIntArray(0, shapeLength);
        int flattenedLength = 0;
        for (int dim : dims) {
            flattenedLength += dim;
        }
        long[] flattenedShapes = data.getValue().getPointer(0).getLongArray(0, flattenedLength);
        int idx = 0;
        List<Shape> result = new ArrayList<>();
        for (int dim : dims) {
            long[] shape = new long[dim];
            System.arraycopy(flattenedShapes, idx, shape, 0, dim);
            idx += dim;
            result.add(new Shape(shape));
        }
        return result;
    }

    public static List<List<Shape>> inferShape(Symbol symbol, PairList<String, Shape> args) {
        Pointer handler = symbol.getHandle();
        int numArgs = args.size();
        String[] keys = args.keys().toArray(new String[0]);
        // the following two is also the representation of
        // CSR NDArray
        long[] indPtr = new long[numArgs + 1];
        Shape flattened = new Shape();
        indPtr[0] = 0;
        for (int i = 0; i < args.size(); i++) {
            Shape shape = args.valueAt(i);
            indPtr[i + 1] = shape.dimension();
            flattened = flattened.addAll(shape);
        }
        long[] flattenedShapeArray = flattened.getShape();

        NativeSizeByReference inShapeSize = new NativeSizeByReference();
        PointerByReference inShapeNDim = REFS.acquire();
        PointerByReference inShapeData = REFS.acquire();
        NativeSizeByReference outShapeSize = new NativeSizeByReference();
        PointerByReference outShapeNDim = REFS.acquire();
        PointerByReference outShapeData = REFS.acquire();
        NativeSizeByReference auxShapeSize = new NativeSizeByReference();
        PointerByReference auxShapeNDim = REFS.acquire();
        PointerByReference auxShapeData = REFS.acquire();
        IntBuffer complete = IntBuffer.allocate(1);
        checkCall(
                LIB.MXSymbolInferShape64(
                        handler,
                        numArgs,
                        keys,
                        indPtr,
                        flattenedShapeArray,
                        inShapeSize,
                        inShapeNDim,
                        inShapeData,
                        outShapeSize,
                        outShapeNDim,
                        outShapeData,
                        auxShapeSize,
                        auxShapeNDim,
                        auxShapeData,
                        complete));
        if (complete.get() != 0) {
            return Arrays.asList(
                    recoverShape(inShapeSize, inShapeNDim, inShapeData),
                    recoverShape(outShapeSize, outShapeNDim, outShapeData),
                    recoverShape(auxShapeSize, auxShapeNDim, auxShapeData));
        }
        return null;
    }

    public static Pointer optimizeFor(Symbol current, String backend, Device device) {
        // TODO: Support partition on parameters
        PointerByReference returnedSymbolHandle = REFS.acquire();
        // placeHolders
        PointerByReference[] placeHolders = {
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire()
        };
        // there is no need to update parameters
        // TODO : check 22th parameter type
        checkCall(
                LIB.MXOptimizeForBackend(
                        current.getHandle(),
                        backend,
                        DeviceType.toDeviceType(device),
                        returnedSymbolHandle,
                        0,
                        placeHolders[0],
                        0,
                        placeHolders[1],
                        0,
                        new String[0],
                        new String[0],
                        0,
                        new String[0],
                        new long[0],
                        new int[0],
                        0,
                        new String[0],
                        new int[0],
                        0,
                        new String[0],
                        new int[0],
                        (byte) 0,
                        IntBuffer.allocate(0),
                        placeHolders[2],
                        placeHolders[3],
                        IntBuffer.allocate(0),
                        placeHolders[4],
                        placeHolders[5]));
        Pointer ptr = returnedSymbolHandle.getValue();
        REFS.recycle(returnedSymbolHandle);
        Arrays.stream(placeHolders).forEach(REFS::recycle);
        return ptr;
    }

    public static String getSymbolString(Pointer symbol) {
        String[] holder = new String[1];
        checkCall(LIB.MXSymbolSaveToJSON(symbol, holder));
        return holder[0];
    }

    public static Pointer getSymbolOutput(Pointer symbol, int index) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolGetOutput(symbol, index, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    /**
     * ***************************************************************************** About NdArray
     * ****************************************************************************
     */
    public static NDList loadNdArray(MxResource parent, Path path, Device device) {
        IntBuffer handlesSize = IntBuffer.allocate(1);
        PointerByReference handlesRef = REFS.acquire();
        PointerByReference namesRef = REFS.acquire();
        IntBuffer namesSize = IntBuffer.allocate(1);
        checkCall(LIB.MXNDArrayLoad(path.toString(), handlesSize, handlesRef, namesSize, namesRef));
        int ndArrayCount = handlesSize.get();
        int nameCount = namesSize.get();
        if (nameCount > 0 && ndArrayCount != nameCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + path.toString());
        }
        Pointer[] handles = handlesRef.getValue().getPointerArray(0, ndArrayCount);
        NDList ndList = new NDList();
        if (nameCount == 0) {
            for (Pointer handle : handles) {
                ndList.add(NDArray.create(parent, handle));
            }
        } else {
            String[] names = namesRef.getValue().getStringArray(0, nameCount);
            for (int i = 0; i < ndArrayCount; i++) {
                NDArray array = NDArray.create(parent, handles[i]);
                array.setName(names[i]);
                ndList.add(array);
            }
        }

        REFS.recycle(namesRef);
        REFS.recycle(handlesRef);

        // MXNet always load NDArray on CPU
        if (Device.cpu().equals(device)) {
            return ndList;
        }

        NDList ret = ndList.toDevice(device, true);
        ndList.close();
        return ret;
    }

    public static PairList<String, Pointer> loadNdArrayFromFile(String path) {
        IntBuffer handleSize = IntBuffer.allocate(1);
        IntBuffer namesSize = IntBuffer.allocate(1);
        PointerByReference handlesRef = REFS.acquire();
        PointerByReference namesRef = REFS.acquire();
        checkCall(LIB.MXNDArrayLoad(path, handleSize, handlesRef, namesSize, namesRef));
        // TODO : construct NDArray Objects
        int handleCount = handleSize.get();
        int nameCount = namesSize.get();
        if (nameCount > 0 && nameCount != handleCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + path);
        }
        Pointer[] handles = handlesRef.getValue().getPointerArray(0, handleCount);

        PairList<String, Pointer> pairList = new PairList<>();

        if (nameCount == 0) {
            for (Pointer handle : handles) {
                pairList.add(null, handle);
            }
        } else {
            String[] names = namesRef.getValue().getStringArray(0, nameCount);
            for (int i = 0; i < handleCount; i++) {
                pairList.add(names[i], handles[i]);
            }
        }
        REFS.recycle(namesRef);
        REFS.recycle(handlesRef);

        return pairList;
    }

    public static void freeNdArray(Pointer handle) {
        checkCall(LIB.MXNDArrayFree(handle));
    }

    public static Pointer loadNdArrayFromByteArray(byte[] buf, int offset, int size) {
        Memory memory = new Memory(size);
        memory.write(0, buf, offset, size);
        PointerByReference outRef = REFS.acquire();
        checkCall(LIB.MXNDArrayLoadFromRawBytes(memory, new NativeSize(size), outRef));
        Pointer p = outRef.getValue();
        //        outRef.getValue().getPointerArray(0, size);

        REFS.recycle(outRef);
        return p;
    }

    public static Pointer loadNdArrayFromByteBuffer(ByteBuffer byteBuffer) {
        //        Pointer handle = new Pointer(byteBuffer.address);
        //        ((DirectByteBuffer) byteBuffer).address()
        // TODO
        byte[] bytes = new byte[byteBuffer.limit()];
        byteBuffer.get(bytes);
        return loadNdArrayFromByteArray(bytes, 0, byteBuffer.limit());
    }

    public static ByteBuffer saveNdArrayAsByteBuffer(Pointer ndArray) {
        NativeSizeByReference size = new NativeSizeByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArraySaveRawBytes(ndArray, size, ref));
        return ref.getValue().getByteBuffer(0, size.getValue().longValue());
    }

    public static byte[] saveNdArrayAsByteArray(Pointer ndArray) {
        ByteBuffer buffer = saveNdArrayAsByteBuffer(ndArray);
        byte[] bytes = new byte[buffer.limit()];
        buffer.get(bytes);
        return bytes;
    }

    public static void syncCopyToCPU(Pointer ndArray, Pointer data, int len) {
        NativeSize size = new NativeSize(len);
        checkNDArray(ndArray, "copy from");
        checkNDArray(data, "copy to");
        checkCall(LIB.MXNDArraySyncCopyToCPU(ndArray, data, size));
    }

    public static void syncCopyFromCPU(Pointer ndArray, Buffer data, int len) {
        NativeSize size = new NativeSize(len);
        Pointer pointer = Native.getDirectBufferPointer(data);
        checkCall(LIB.MXNDArraySyncCopyFromCPU(ndArray, pointer, size));
    }

    public static void waitToRead(Pointer ndArray) {
        checkNDArray(ndArray, "wait to read");
        checkCall(LIB.MXNDArrayWaitToRead(ndArray));
    }

    public static void waitToWrite(Pointer ndArray) {
        checkNDArray(ndArray, "wait to write");
        checkCall(LIB.MXNDArrayWaitToWrite(ndArray));
    }

    public static Pointer detachGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXNDArrayDetach(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer getGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the gradient for");
        checkCall(LIB.MXNDArrayGetGrad(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static void autogradMarkVariables(
            int numVar, Pointer varHandles, IntBuffer reqsArray, Pointer gradHandles) {
        PointerByReference varRef = REFS.acquire();
        PointerByReference gradRef = REFS.acquire();
        varRef.setValue(varHandles);
        gradRef.setValue(gradHandles);
        checkCall(LIB.MXAutogradMarkVariables(numVar, varRef, reqsArray, gradRef));
        REFS.recycle(varRef);
        REFS.recycle(gradRef);
    }

    public static Map<String, FunctionInfo> getNdArrayFunctions() {
        Set<String> opNames = JnaUtils.getAllOpNames();
        Map<String, FunctionInfo> map = new ConcurrentHashMap<>();

        PointerByReference ref = REFS.acquire();
        for (String opName : opNames) {
            checkCall(LIB.NNGetOpHandle(opName, ref));

            String functionName = getOpNamePrefix(opName);

            // System.out.println("Name: " + opName + "/" + functionName);
            map.put(functionName, getFunctionByName(opName, functionName, ref.getValue()));
            ref.setValue(null);
        }
        REFS.recycle(ref);
        return map;
    }

    public static PairList<Pointer, SparseFormat> imperativeInvoke(
            Pointer function, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        String[] keys;
        String[] values;
        if (params == null) {
            keys = EMPTY_ARRAY;
            values = EMPTY_ARRAY;
        } else {
            keys = params.keyArray(EMPTY_ARRAY);
            values = params.values().stream().map(Object::toString).toArray(String[]::new);
        }
        //        StringArray keyArray = StringArray.of(keys);
        //        StringArray valueArray = StringArray.of(values);
        PointerArray srcArray = toPointerArray(src);
        PointerArray destArray = toPointerArray(dest);
        PointerByReference destRef = REFS.acquire();
        destRef.setValue(destArray);
        PointerByReference destSType = REFS.acquire();
        IntBuffer numOutputs = IntBuffer.allocate(1);
        numOutputs.put(0, 1);

        checkCall(
                LIB.MXImperativeInvoke(
                        function,
                        src.length,
                        srcArray,
                        numOutputs,
                        destRef,
                        keys.length,
                        keys,
                        values,
                        destSType));
        int numOfOutputs = numOutputs.get(0);
        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOfOutputs);
        int[] sTypes = destSType.getValue().getIntArray(0, numOfOutputs);
        PairList<Pointer, SparseFormat> pairList = new PairList<>();
        for (int i = 0; i < numOfOutputs; i++) {
            pairList.add(ptrArray[i], SparseFormat.fromValue(sTypes[i]));
        }
        REFS.recycle(destRef);
        REFS.recycle(destSType);
        srcArray.recycle();
        //        keyArray.recycle();
        //        valueArray.recycle();

        if (destArray != null) {
            destArray.recycle();
        }
        return pairList;
    }

    private static PointerArray toPointerArray(NDArray[] vals) {
        if (vals == null) {
            return null;
        }
        Pointer[] valPointers = new Pointer[vals.length];
        for (int i = 0; i < vals.length; i++) {
            valPointers[i] = vals[i].getHandle();
        }
        return PointerArray.of(valPointers);
    }

    public static FunctionInfo op(String opName) {
        if (!OPS.containsKey(opName)) {
            throw new IllegalArgumentException("Unknown operator: " + opName);
        }
        return OPS.get(opName);
    }

    public static FunctionInfo getFunctionByName(String name, String functionName, Pointer handle) {
        String[] nameRef = {name};
        String[] description = new String[1];
        IntBuffer numArgs = IntBuffer.allocate(1);
        PointerByReference argNameRef = REFS.acquire();
        PointerByReference argTypeRef = REFS.acquire();
        PointerByReference argDescRef = REFS.acquire();
        String[] keyVarArgs = new String[1];
        String[] returnType = new String[1];

        checkCall(
                LIB.MXSymbolGetAtomicSymbolInfo(
                        handle,
                        nameRef,
                        description,
                        numArgs,
                        argNameRef,
                        argTypeRef,
                        argDescRef,
                        keyVarArgs,
                        returnType));

        int count = numArgs.get();
        PairList<String, String> arguments = new PairList<>();
        if (count != 0) {
            String[] argNames =
                    argNameRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            String[] argTypes =
                    argTypeRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            for (int i = 0; i < argNames.length; i++) {
                arguments.add(argNames[i], argTypes[i]);
            }
        }

        REFS.recycle(argNameRef);
        REFS.recycle(argTypeRef);
        REFS.recycle(argDescRef);

        return new FunctionInfo(handle, functionName, arguments);
    }

    public static Set<String> getAllOpNames() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference outArray = REFS.acquire();

        checkCall(LIB.MXListAllOpNames(outSize, outArray));

        int size = outSize.get();
        Pointer[] pointers = outArray.getValue().getPointerArray(0, size);

        Set<String> set = new HashSet<>();
        for (Pointer p : pointers) {
            set.add(p.getString(0, StandardCharsets.UTF_8.name()));
        }
        REFS.recycle(outArray);
        return set;
    }

    public static String getOpNamePrefix(String name) {
        for (String prefix : OP_NAME_PREFIX) {
            if (name.startsWith(prefix)) {
                return name.substring(prefix.length());
            }
        }
        return name;
    }

    public static DataType getDataTypeOfNdArray(Pointer handle) {
        IntBuffer dataType = IntBuffer.allocate(1);
        checkNDArray(handle, "get the data type of");
        checkCall(LIB.MXNDArrayGetDType(handle, dataType));
        return DataType.values()[dataType.get()];
    }

    public static Device getDeviceOfNdArray(Pointer handle) {
        IntBuffer deviceType = IntBuffer.allocate(1);
        IntBuffer deviceId = IntBuffer.allocate(1);
        checkNDArray(handle, "get the device of");
        checkCall(LIB.MXNDArrayGetContext(handle, deviceType, deviceId));
        String deviceTypeStr = DeviceType.fromDeviceType(deviceType.get(0));
        // CPU is special case which don't have device id
        return Device.of(deviceTypeStr, deviceId.get(0));
    }

    public static Shape getShapeOfNdArray(Pointer handle) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the shape of");
        checkCall(LIB.MXNDArrayGetShape(handle, dim, ref));
        int nDim = dim.get();
        if (nDim == 0) {
            REFS.recycle(ref);
            return new Shape();
        }
        int[] shape = ref.getValue().getIntArray(0, nDim);
        REFS.recycle(ref);
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static Shape getShape64OfNdArray(Pointer handle) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the shape64 of");
        checkCall(LIB.MXNDArrayGetShape64(handle, dim, ref));
        int nDim = dim.get();
        if (nDim == 0) {
            REFS.recycle(ref);
            return new Shape();
        }
        int[] shape = ref.getValue().getIntArray(0, nDim);
        REFS.recycle(ref);
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static SparseFormat getStorageType(Pointer handle) {
        IntBuffer type = IntBuffer.allocate(1);
        checkNDArray(handle, "get the storage type of");
        checkCall(LIB.MXNDArrayGetStorageType(handle, type));
        return SparseFormat.fromValue(type.get());
    }

    public static Pointer createNdArray(
            Device device, Shape shape, DataType dataType, int size, boolean delayedAlloc) {
        int deviceType = DeviceType.toDeviceType(device);
        int deviceId = (deviceType != 1) ? device.getDeviceId() : -1;
        int delay = delayedAlloc ? 1 : 0;

        PointerByReference ref = REFS.acquire();
        int[] shapeArray = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        checkCall(
                LIB.MXNDArrayCreate(
                        shapeArray, size, deviceType, deviceId, delay, dataType.ordinal(), ref));

        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer createSparseNdArray(
            SparseFormat fmt,
            Device device,
            Shape shape,
            DataType dtype,
            DataType[] auxDTypes,
            Shape[] auxShapes,
            boolean delayedAlloc) {
        int[] shapeArray = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        int deviceType = DeviceType.toDeviceType(device);
        int deviceId = (deviceType != 1) ? device.getDeviceId() : -1;
        int delay = delayedAlloc ? 1 : 0;
        PointerByReference ref = REFS.acquire();
        IntBuffer auxDTypesInt =
                IntBuffer.wrap(Arrays.stream(auxDTypes).mapToInt(DataType::ordinal).toArray());
        IntBuffer auxNDims =
                IntBuffer.wrap(Arrays.stream(auxShapes).mapToInt(Shape::dimension).toArray());
        int[] auxShapesInt = Arrays.stream(auxShapes).mapToInt(ele -> (int) ele.head()).toArray();
        checkCall(
                LIB.MXNDArrayCreateSparseEx(
                        fmt.getValue(),
                        shapeArray,
                        shapeArray.length,
                        deviceType,
                        deviceId,
                        delay,
                        dtype.ordinal(),
                        auxDTypes.length,
                        auxDTypesInt,
                        auxNDims,
                        auxShapesInt,
                        ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static void ndArraySyncCopyFromNdArray(NDArray dest, NDArray src, int location) {
        checkCall(LIB.MXNDArraySyncCopyFromNDArray(dest.getHandle(), src.getHandle(), location));
    }

    public static int getVersion() {
        IntBuffer version = IntBuffer.allocate(1);
        checkCall(LIB.MXGetVersion(version));
        return version.get();
    }

    public static NDArray[] cachedOpInvoke(
            MxResource parent, Pointer cachedOpHandle, NDArray[] inputs) {
        IntBuffer buf = IntBuffer.allocate(1);
        PointerArray array = toPointerArray(inputs);
        PointerByReference ref = REFS.acquire();
        PointerByReference outSTypeRef = REFS.acquire();
        Device device = inputs[0].getDevice();
        // TODO: check the init value of default_dev_type and default_dev_id
        checkCall(
                LIB.MXInvokeCachedOp(
                        cachedOpHandle,
                        inputs.length,
                        array,
                        DeviceType.toDeviceType(device),
                        0,
                        buf,
                        ref,
                        outSTypeRef));
        int numOutputs = buf.get();
        Pointer[] ptrArray = ref.getValue().getPointerArray(0, numOutputs);
        int[] sTypes = outSTypeRef.getValue().getIntArray(0, numOutputs);
        NDArray[] output = new NDArray[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            if (sTypes[i] != 0) {
                output[i] = NDArray.create(parent, ptrArray[i], SparseFormat.fromValue(sTypes[i]));
            } else {
                output[i] = NDArray.create(parent, ptrArray[i]);
            }
        }
        REFS.recycle(ref);
        REFS.recycle(outSTypeRef);
        array.recycle();
        return output;
    }

    private static void checkNDArray(Pointer pointer, String msg) {
        if (pointer == null) {
            throw new IllegalArgumentException(
                    "Tried to " + msg + " an MXNet NDArray that was already closed");
        }
    }

    public static void checkCall(int ret) {
        if (ret != 0) {
            logger.error("MXNet engine call failed: " + getLastError());
            throw new JnaCallException("MXNet engine call failed: " + getLastError());
        }
    }

    private static String getLastError() {
        return LIB.MXGetLastError();
    }

    private static String[] toStringArray(PointerByReference ref, int size) {
        if (size == 0) {
            return new String[0];
        }

        Pointer[] pointers = ref.getValue().getPointerArray(0, size);

        String[] arr = new String[size];
        for (int i = 0; i < size; ++i) {
            arr[i] = pointers[i].getString(0, StandardCharsets.UTF_8.name());
        }

        return arr;
    }

    /**
     * *************************************************************************** Others
     * ***************************************************************************
     */
    private static Set<String> getFeaturesInternal() {
        PointerByReference ref = REFS.acquire();
        NativeSizeByReference outSize = new NativeSizeByReference();
        checkCall(LIB.MXLibInfoFeatures(ref, outSize));

        int size = outSize.getValue().intValue();
        if (size == 0) {
            REFS.recycle(ref);
            return Collections.emptySet();
        }

        LibFeature pointer = new LibFeature(ref.getValue());
        pointer.read();

        LibFeature[] features = (LibFeature[]) pointer.toArray(size);

        Set<String> set = new HashSet<>();
        for (LibFeature feature : features) {
            if (feature.getEnabled() == 1) {
                set.add(feature.getName());
            }
        }
        REFS.recycle(ref);
        return set;
    }

    public static Set<String> getFeatures() {
        return FEATURES;
    }

    public static boolean autogradIsTraining() {
        ByteBuffer isTraining = ByteBuffer.allocate(1);
        checkCall(LIB.MXAutogradIsTraining(isTraining));
        return isTraining.get(0) == 1;
    }
}
