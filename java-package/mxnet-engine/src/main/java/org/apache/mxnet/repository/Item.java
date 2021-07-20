package org.apache.mxnet.repository;

public enum Item {
    MLP("mlp", "https://resources.djl.ai/test-models/mlp.tar.gz");

    private String name;
    private String url;


    Item(String name, String url) {
        this.name = name;
        this.url = url;
    }

    public String getName() {
        return name;
    }

    public String getUrl() {
        return url;
    }
}
