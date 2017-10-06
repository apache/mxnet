package io.mxnet.caffetranslator;

public class GeneratorOutput {
    public String code;
    public int numLayersTranslated;

    public GeneratorOutput(String code, int n) {
        this.code = code;
        this.numLayersTranslated = n;
    }
}
