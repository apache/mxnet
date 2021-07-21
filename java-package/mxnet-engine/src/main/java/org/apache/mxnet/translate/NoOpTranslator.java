package org.apache.mxnet.translate;

import org.apache.mxnet.ndarray.MxNDList;

public class NoOpTranslator implements Translator<MxNDList, MxNDList>{

    public NoOpTranslator() {}

    @Override
    public Pipeline getPipeline() {
        return Translator.super.getPipeline();
    }

    @Override
    public MxNDList processInput(MxNDList input) {
        return input;
    }

    @Override
    public MxNDList processOutput(MxNDList output) {
        return output;
    }
}
