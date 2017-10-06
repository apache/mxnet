package io.mxnet.caffetranslator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SolverListener extends CaffePrototxtBaseListener {

    Map<String, List<String>> properties;
    ParserHelper parserHelper;

    public SolverListener() {
        properties = new HashMap<>();
        parserHelper = new ParserHelper();
    }

    public Map<String, List<String>> getProperties() {
        return properties;
    }

    @Override
    public void exitPair(CaffePrototxtParser.PairContext ctx) {
        String key = ctx.ID().getText();
        String value = ctx.value().getText();
        value = parserHelper.removeQuotes(value);

        if (properties.get(key) == null)
            properties.put(key, new ArrayList<>());

        properties.get(key).add(value);
    }
}
