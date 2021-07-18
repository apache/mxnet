package org.apache.mxnet.util;

import com.sun.jna.Pointer;
import java.util.concurrent.atomic.AtomicReference;

public abstract class NativeResource<T> implements AutoCloseable {

    protected final AtomicReference<T> handle;
    private String uid;

    protected NativeResource(T handle) {
        this.handle = new AtomicReference<>(handle);
        this.uid = handle.toString();
    }

    protected NativeResource() {
        this.handle = null;
        this.uid = null;
    }

    /**
     * To initialize a NativeResource with handle = null
     * @param uid
     */
    protected NativeResource(String uid) {
        this.handle = null;
        this.uid = uid;
    }

    /**
     * Gets the boolean that indicates whether this resource has been released.
     *
     * @return whether this resource has been released
     */
    public boolean isReleased() {
        return handle.get() == null;
    }

    /**
     * Gets the {@link Pointer} to this resource.
     *
     * @return the {@link Pointer} to this resource
     */
    public T getHandle() {
        T reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return reference;
    }

    /**
     * Gets the unique ID of this resource.
     *
     * @return the unique ID of this resource
     */
    public final String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
