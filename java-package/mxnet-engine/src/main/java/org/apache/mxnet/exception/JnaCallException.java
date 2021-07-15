package org.apache.mxnet.exception;

public class JnaCallException extends BaseException {

    private static final long serialVersionUID = 1L;

    public JnaCallException(String message) {
        super(message);
    }

    public JnaCallException(String message, Throwable cause) {
        super(message, cause);
    }

    public JnaCallException(Throwable cause) {
        super(cause);
    }
}
