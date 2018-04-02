/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file StatsListener.java
 * \brief ANTLR listener to collect stats used by CollectStats.java
 */

package io.mxnet.caffetranslator.misc;

import io.mxnet.caffetranslator.CaffePrototxtBaseListener;
import io.mxnet.caffetranslator.CaffePrototxtParser;
import io.mxnet.caffetranslator.ParserHelper;
import lombok.Getter;

import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

public class StatsListener extends CaffePrototxtBaseListener {

    private final Stack<String> keys;
    @Getter
    private final Map<String, Set<String>> attrMap;
    private final ParserHelper parserHelper;

    private String layerType;
    private Set<String> curAttr;

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
        if (key.equals("type")) {
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
