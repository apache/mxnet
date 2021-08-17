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

package org.apache.mxnet.util;

import com.sun.jna.Pointer;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@code NativeResource} is an internal class for {@link AutoCloseable} blocks of memory.
 *
 * @param <T> the resource that could map to a native pointer or java object
 */
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
     * To initialize a NativeResource with handle = null.
     *
     * @param uid for the {@link NativeResource}
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
