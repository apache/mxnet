package org.apache.mxnet.engine;

import com.sun.jna.Pointer;

public class BaseMxResource extends MxResource{

    static BaseMxResource SYSTEM_MX_RESOURCE = new BaseMxResource();

    protected BaseMxResource(Pointer handle) {
        super(handle);
    }

    protected BaseMxResource() {
        super();
    }

    public static BaseMxResource getSystemMxResource() {
        return SYSTEM_MX_RESOURCE;
    }

    public boolean isReleased() {
        return handle.get() == null;
    }
}
