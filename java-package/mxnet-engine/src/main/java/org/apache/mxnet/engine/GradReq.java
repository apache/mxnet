package org.apache.mxnet.engine;

/** An enum that indicates whether gradient is required. */
public enum GradReq {
    NULL("null", 0),
    WRITE("write", 1),
    ADD("add", 3);

    private String type;
    private int value;

    GradReq(String type, int value) {
        this.type = type;
        this.value = value;
    }

    /**
     * Gets the type of this {@code GradReq}.
     *
     * @return the type
     */
    public String getType() {
        return type;
    }

    /**
     * Gets the value of this {@code GradType}.
     *
     * @return the value
     */
    public int getValue() {
        return value;
    }
}