package io.mxnet.caffetranslator;

import java.util.List;
import java.util.Vector;

public class Config {

    private static final Config instance = new Config();

    public static Config getInstance() {
        return instance;
    }

    private Config() {
        if (instance != null) {
            throw new IllegalStateException("Already instantiated");
        }

        customDataLayers = new Vector<String>();
    }

    public List<String> getCustomDataLayers() {
        return customDataLayers;
    }

    public void addCustomDataLayer(String name) {
        customDataLayers.add(name);
    }

    private Vector<String> customDataLayers;
}
