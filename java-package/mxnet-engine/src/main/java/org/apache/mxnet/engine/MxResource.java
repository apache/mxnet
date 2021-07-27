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
import org.apache.mxnet.util.NativeResource;

import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class MxResource extends NativeResource<Pointer> {

    public static final String EMPTY_UID = "EMPTY_UID";

    private static boolean closed = false;

    protected Device device = null;

    public void setClosed() {
        this.closed = true;
    }

    public boolean getClosed() {
        return closed;
    }

    protected MxResource(MxResource parent, String uid) {
        super(uid);
        setParent(parent);
        getParent().addSubResource(this);
    }

    // initial a MxResource object with random uid
    protected MxResource(MxResource parent) {
        this(parent, UUID.randomUUID().toString());
    }

//    protected static MxResource createEmptyMxResource(MxResource parent) {
//        return new MxResource(parent, EMPTY_UID);
//    }

    private MxResource parent = null;

    private ConcurrentHashMap<String, MxResource> subResources = null;

    public void addSubResource(MxResource subMxResource) {
        getSubResource().put(subMxResource.getUid(), subMxResource);
    }

    public void freeSubResources() {
        if (subResourceInitialized()) {
            for (MxResource e : subResources.values()) {
                if (!e.getClosed()) {
                    e.close();
                }
            }
            subResources = null;
        }
    }

    public boolean subResourceInitialized() {
        return subResources != null;
    }

    public ConcurrentHashMap<String, MxResource> getSubResource() {
        if (!subResourceInitialized()) {
            subResources = new ConcurrentHashMap<>();
        }
        return subResources;
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

    protected void setParent(MxResource parent) {
        this.parent = parent;
    }

    public MxResource getParent() {
        return this.parent;
    }

    protected MxResource() {
        super();
        setParent(null);
    }

    public void setDevice(Device device) {
        this.device = device;
    }

    public Device getDevice() {
        Device device = getParent() == null ? null : getParent().getDevice();
        return Device.defaultIfNull(device);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        freeSubResources();
        setClosed();
    }

}
