package org.apache.mxnet.engine;

public class EmptyMxResource extends MxResource {

    private EmptyMxResource() {
        super();
    }

    public EmptyMxResource(MxResource parent) {
        super(parent, EMPTY_UID);
    }

}
