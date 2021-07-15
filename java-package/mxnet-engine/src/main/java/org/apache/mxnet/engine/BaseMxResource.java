package org.apache.mxnet.engine;

public class BaseMxResource extends MxResource{

    static BaseMxResource SYSTEM_MX_RESOURCE = new BaseMxResource();

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
