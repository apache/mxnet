package org.apache.mxnet.api.exception;

public class EngineException extends BaseException{

    private static final long serialVersionUID = 1L;

    public EngineException(String message) {
        super(message);
    }

    public EngineException(String message, Throwable cause) {
        super(message, cause);
    }

    public EngineException(Throwable cause) {
        super(cause);
    }
}
