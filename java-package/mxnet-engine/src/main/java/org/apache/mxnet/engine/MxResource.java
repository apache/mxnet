package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.util.NativeResource;
import java.util.concurrent.ConcurrentHashMap;

public class MxResource extends NativeResource<Pointer> {

    private MxResource parent = null;

    private ConcurrentHashMap<String, MxResource> subResources;

    public void addSubResource(MxResource subResource) {
        subResources.put(subResource.getUid(), subResource);
    }

    public void freeSubResources() {
        subResources.values().stream().forEach(MxResource::close);
        subResources = null;
    }

    protected MxResource(MxResource parent, Pointer handle) {
        super(handle);
        setParent(parent);
        this.subResources = new ConcurrentHashMap<>();
        if (parent != null) {
            parent.addSubResource(this);
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
        this.subResources = new ConcurrentHashMap<>();
    }

}
