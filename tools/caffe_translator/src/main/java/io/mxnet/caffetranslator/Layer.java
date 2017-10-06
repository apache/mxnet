package io.mxnet.caffetranslator;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Layer {

    @Getter
    @Setter
    private String name;

    @Getter
    @Setter
    private int layerIndex;

    @Getter
    @Setter
    private Kind kind;

    @Getter
    private List<String> bottoms;

    @Getter
    private List<String> tops;

    @Setter
    @Getter
    private List<Map<String, String>> params;

    @Getter
    @Setter
    private String prototxt;

    @Setter
    private Map<String, List<String>> attr;

    public Layer() {
        tops = new ArrayList<>();
        bottoms = new ArrayList<>();
        attr = new HashMap<>();
        params = new ArrayList<>();
    }

    public Layer(int layerIndex) {
        this();
        this.layerIndex = layerIndex;
    }

    public void addAttr(String key, String value) {
        List<String> list = attr.get(key);
        if (list == null) {
            list = new ArrayList<String>();
            list.add(value);
            attr.put(key, list);
        } else {
            list.add(value);
        }
    }

    public String getAttr(String key) {
        List<String> list = attr.get(key);
        if (list == null) return null;

        return list.get(0);
    }

    public String getAttr(String key, String defaultValue) {
        String attr = getAttr(key);
        return attr != null ? attr : defaultValue;
    }

    public boolean hasAttr(String key) {
        return attr.containsKey(key);
    }

    public boolean attrEquals(String key, String value) {
        if (!attr.containsKey(key)) return false;
        return getAttr(key).equals(value);
    }

    public List<String> getAttrList(String key) {
        return attr.get(key);
    }

    public void addTop(String top) {
        tops.add(top);
    }

    public void addBottom(String bottom) {
        bottoms.add(bottom);
    }

    public String getBottom() {
        return bottoms.size() > 0 ? bottoms.get(0) : null;
    }

    public String getType() {
        return attr.get(Constants.TYPE).get(0);
    }

    public String getTop() {
        return tops.size() > 0 ? tops.get(0) : null;
    }

    public enum Kind {
        DATA, INTERMEDIATE, LOSS;
    }
}
