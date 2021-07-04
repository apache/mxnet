package org.apache.mxnet.jna;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import org.apache.mxnet.api.exception.EngineException;

public final class JnaUtils {
    public static final MxnetLibrary LIB = LibUtils.loadLibrary();
    public static final ObjectPool<PointerByReference> REFS =
            new ObjectPool<>(PointerByReference::new, r -> r.setValue(null));

    //    private static final Map<String, FunctionInfo> OPS = getNdArrayFunctions();
    // TODO

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

    private static String getLastError() {
        return LIB.MXGetLastError();
    }

    public static void checkCall(int ret) {
        if (ret != 0) {
            throw new EngineException("MXNet engine call failed: " + getLastError());
        }
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

    public static void main(String... args) {
        System.getenv();
        try {
            Pointer p =
                    JnaUtils.createSymbolFromFile(
                            "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json");
            String[] output = listSymbolOutputs(p);
            String strSymbol = printSymbol(p);
            System.out.println("catch");
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }
}
