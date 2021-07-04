package org.apache.mxnet.jna;

import com.sun.jna.Pointer;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A FunctionInfo represents an operator (ie function) within the MXNet Engine. */
public class FunctionInfo {

    private Pointer handle;
    private String name;
    private PairList<String, String> arguments;

    private static final Logger logger = LoggerFactory.getLogger(FunctionInfo.class);

    FunctionInfo(Pointer pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    // TODO : complete other methods

}
