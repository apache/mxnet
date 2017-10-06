package io.mxnet.caffetranslator;

import java.util.HashMap;
import java.util.Map;

public class SymbolGeneratorFactory {

    private static SymbolGeneratorFactory instance = new SymbolGeneratorFactory();
    Map<String, SymbolGenerator> generators;

    public static SymbolGeneratorFactory getInstance() {
        return instance;
    }

    private SymbolGeneratorFactory() {
        if (instance != null) {
            throw new IllegalStateException("SymbolGeneratorFactory already instantiated");
        }
        generators = new HashMap<>();
    }

    public SymbolGenerator getGenerator(String symbolType) {
        return generators.get(symbolType);
    }

    public void addGenerator(String symbolType, SymbolGenerator generator) {
        generators.put(symbolType, generator);
    }
}
