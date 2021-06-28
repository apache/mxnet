package org.apache.mxnet.engine;

import org.apache.mxnet.Device;

public abstract class Engine {

    //TODO

    public static Engine getInstance() {
        return null;
    }

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return {@code true} if the engine has the specified capability
     */
    public abstract boolean hasCapability(String capability);

    /**
     * Returns the engine's default {@link Device}.
     *
     * @return the engine's default {@link Device}
     */
    public Device defaultDevice() {
        return null;
    }

}
