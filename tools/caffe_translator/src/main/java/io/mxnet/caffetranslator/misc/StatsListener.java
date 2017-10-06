package io.mxnet.caffetranslator.misc;

import io.mxnet.caffetranslator.CaffePrototxtBaseListener;
import io.mxnet.caffetranslator.CaffePrototxtParser;
import io.mxnet.caffetranslator.Constants;
import io.mxnet.caffetranslator.ParserHelper;
import lombok.Getter;

import java.util.*;

public class StatsListener extends CaffePrototxtBaseListener {

    private String layerType;
    private Stack<String> keys;
    @Getter
    private Map<String, Set<String>> attrMap;
    private Set<String> curAttr;
    ParserHelper parserHelper;

    public StatsListener() {
        attrMap = new TreeMap<>();
        keys = new Stack<>();
        parserHelper = new ParserHelper();
    }

    @Override
    public void enterLayer(CaffePrototxtParser.LayerContext ctx) {
        keys.clear();
        curAttr = new TreeSet<>();
    }

    @Override
    public void exitLayer(CaffePrototxtParser.LayerContext ctx) {
        if (!attrMap.containsKey(layerType)) {
            attrMap.put(layerType, new TreeSet<>());
        }
        Set<String> set = attrMap.get(layerType);
        set.addAll(curAttr);
    }

    @Override
    public void exitValueLeaf(CaffePrototxtParser.ValueLeafContext ctx) {
        String value = ctx.getText();
        value = parserHelper.removeQuotes(value);
        processKeyValue(getCurrentKey(), value);
    }

    private void processKeyValue(String key, String value) {
        if (key.equals(Constants.TYPE)) {
            layerType = value;
        } else {
            curAttr.add(key);
        }
    }

    @Override
    public void enterPair(CaffePrototxtParser.PairContext ctx) {
        String key = ctx.getStart().getText();
        keys.push(key);
    }

    @Override
    public void exitPair(CaffePrototxtParser.PairContext ctx) {
        keys.pop();
    }

    private String getCurrentKey() {
        StringBuilder sb = new StringBuilder();
        for (String s : keys) {
            sb.append(s + ".");
        }
        return sb.substring(0, sb.length() - 1).toString();
    }

}
