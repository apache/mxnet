package org.apache.mxnet.engine;

public final class BaseMxResource extends MxResource{

    static BaseMxResource SYSTEM_MX_RESOURCE;

    protected BaseMxResource() {
        super();
    }

    public static BaseMxResource getSystemMxResource() {
        if (SYSTEM_MX_RESOURCE == null) {
            SYSTEM_MX_RESOURCE = new BaseMxResource();
        }
        return SYSTEM_MX_RESOURCE;
    }

    public static MxResource newSubMxResource() {
        return new MxResource(getSystemMxResource());
    }

    public boolean isReleased() {
        return handle.get() == null;
    }

    @Override
    public void close() {
        // only clean sub resources
        super.close();
    }
}
