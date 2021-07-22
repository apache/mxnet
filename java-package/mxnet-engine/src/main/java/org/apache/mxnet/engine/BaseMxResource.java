package org.apache.mxnet.engine;

import org.apache.mxnet.jna.JnaUtils;

public final class BaseMxResource extends MxResource{

    private static BaseMxResource SYSTEM_MX_RESOURCE;

    protected BaseMxResource() {
        super();
        // Workaround MXNet engine lazy initialization issue
        JnaUtils.getAllOpNames();

        JnaUtils.setNumpyMode(JnaUtils.NumpyMode.GLOBAL_ON);

        // Workaround MXNet shutdown crash issue
        Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD
    }

    public static BaseMxResource getSystemMxResource() {
        if (SYSTEM_MX_RESOURCE == null) {
            SYSTEM_MX_RESOURCE = new BaseMxResource();
        }
        return SYSTEM_MX_RESOURCE;
    }

//    public static MxResource newSubMxResource() {
//        return new MxResource(getSystemMxResource());
//    }

    public boolean isReleased() {
        return handle.get() == null;
    }

    @Override
    public void close() {
        // only clean sub resources
        super.close();
    }
}
