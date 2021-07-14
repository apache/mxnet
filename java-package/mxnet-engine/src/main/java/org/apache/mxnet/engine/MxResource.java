package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.util.NativeResource;
import java.util.concurrent.ConcurrentHashMap;

public class MxResource extends NativeResource<Pointer> {

    private MxResource parent = null;

    private ConcurrentHashMap<String, NativeResource> subResources;

    public void addSubResource(NativeResource<Pointer> nativeResource) {
        subResources.put(nativeResource.getUid(), nativeResource);
    }

    public void freeSubResources() {
        subResources.values().stream().forEach(NativeResource::close);
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

    protected MxResource getParent() {
        return this.parent;
    }

    protected MxResource() {
        super();
        setParent(null);
        this.subResources = new ConcurrentHashMap<>();
    }

}
