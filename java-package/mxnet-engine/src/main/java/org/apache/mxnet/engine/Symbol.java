package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.util.NativeResource;

// TODO
public class Symbol extends NativeResource<Pointer> {

    private String[] outputs;

    protected Symbol(Pointer handle) {
        super(handle);
    }

    public static Symbol loadFromFile(String path) {
        Pointer p = JnaUtils.createSymbolFromFile(path);
        return new Symbol(p);
    }

    public String[] getOutputNames() {
        if (this.outputs == null) {
            this.outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return this.outputs;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeSymbol(pointer);
        }
    }

    public static void main(String... args) {
        try (Symbol symbol =
                Symbol.loadFromFile(
                        "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json")) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
