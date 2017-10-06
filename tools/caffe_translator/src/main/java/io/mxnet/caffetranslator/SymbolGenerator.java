package io.mxnet.caffetranslator;

public interface SymbolGenerator {
    public GeneratorOutput generate(Layer layer, MLModel model);
}
