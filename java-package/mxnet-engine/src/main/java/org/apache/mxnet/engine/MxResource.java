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

package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.util.NativeResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An auto closable Resource object whose life circle can be managed by its parent {@link
 * MxResource} instance. Meanwhile, it manages life circle of child {@link MxResource} instances.
 */
public class MxResource extends NativeResource<Pointer> {

    private static final Logger logger = LoggerFactory.getLogger(MxResource.class);

    private static boolean closed;

    protected Device device;

    private MxResource parent;

    private ConcurrentHashMap<String, MxResource> subResources;

    protected MxResource() {
        super();
        setParent(null);
    }

    protected MxResource(MxResource parent, String uid) {
        super(uid);
        setClosed(false);
        setParent(parent);
        getParent().addSubResource(this);
    }

    protected MxResource(MxResource parent) {
        this(parent, UUID.randomUUID().toString());
    }

    protected MxResource(MxResource parent, Pointer handle) {
        super(handle);
        setParent(parent);
        if (parent != null) {
            parent.addSubResource(this);
        } else {
            BaseMxResource.getSystemMxResource().addSubResource(this);
        }
    }
    /**
     * Add the sub {@link MxResource} under the current instance.
     *
     * @param subMxResource the instance to be added
     */
    public void addSubResource(MxResource subMxResource) {
        getSubResource().put(subMxResource.getUid(), subMxResource);
    }

    /** Free all sub {@link MxResource} instances of the current instance. */
    public void freeSubResources() {
        if (subResourceInitialized()) {
            for (MxResource subResource : subResources.values()) {
                try {
                    subResource.close();
                } catch (Exception e) {
                    logger.error("MxResource close failed.", e);
                }
            }
            subResources = null;
        }
    }

    /**
     * Check whether {@code subResource} has been initialized.
     *
     * @return boolean
     */
    public boolean subResourceInitialized() {
        return subResources != null;
    }

    /**
     * Get the {@code subResources} of the {@link MxResource}.
     *
     * @return subResources
     */
    public ConcurrentHashMap<String, MxResource> getSubResource() {
        if (!subResourceInitialized()) {
            subResources = new ConcurrentHashMap<>();
        }
        return subResources;
    }

    protected final void setParent(MxResource parent) {
        this.parent = parent;
    }

    /**
     * Get parent {@link MxResource} of the current instance.
     *
     * @return {@link MxResource}
     */
    public MxResource getParent() {
        return this.parent;
    }

    /**
     * Set the {@link Device} for the {@link MxResource}.
     *
     * @param device {@link Device}
     */
    public void setDevice(Device device) {
        this.device = device;
    }

    /**
     * Returns the {@link Device} of this {@code MxResource}.
     *
     * <p>{@link Device} class contains the information where this {@code NDArray} stored in memory,
     * like CPU/GPU.
     *
     * @return the {@link Device} of this {@code MxResource}
     */
    public Device getDevice() {
        Device curDevice = getParent() == null ? null : getParent().getDevice();
        return Device.defaultIfNull(curDevice);
    }

    /**
     * Sets closed for MxResource instance.
     *
     * @param isClosed whether this {@link MxResource} get closed
     */
    public final void setClosed(boolean isClosed) {
        this.closed = isClosed;
    }

    /**
     * Get the attribute closed for the MxResource to check out whether it is closed.
     *
     * @return closed
     */
    public boolean getClosed() {
        return closed;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        freeSubResources();
        setClosed(true);
    }
}
