package org.apache.mxnet.exception;

public class TranslateException extends BaseException{

    private static final long serialVersionUID = 1L;

    public TranslateException(String message) {
        super(message);
    }

    public TranslateException(String message, Throwable cause) {
        super(message, cause);
    }

    public TranslateException(Throwable cause) {
        super(cause);
    }
}
