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
 * \file CreateModelListener.java
 * \brief ANTLR listener that builds MLModel as the parser parses the Caffe prototxt
 */

package io.mxnet.caffetranslator;

import lombok.Getter;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class CreateModelListener extends CaffePrototxtBaseListener {

    private final CaffePrototxtParser parser;
    @Getter
    private final MLModel mlModel;
    private final Stack<String> keys;
    private final ParserHelper parserHelper;

    private Layer currentLayer;
    private Map<String, String> currentParams;

    public CreateModelListener(CaffePrototxtParser parser, MLModel mlModel) {
        this.parser = parser;
        this.mlModel = mlModel;
        this.keys = new Stack<>();
        this.currentParams = new HashMap<>();
        this.parserHelper = new ParserHelper();
    }

    @Override
    public void exitName(CaffePrototxtParser.NameContext ctx) {
        String name = ctx.STRING().toString();
        mlModel.setName(parserHelper.removeQuotes(name));
    }

    @Override
    public void enterLayer(CaffePrototxtParser.LayerContext ctx) {
        keys.clear();
        currentLayer = new Layer();
    }

    @Override
    public void exitLayer(CaffePrototxtParser.LayerContext ctx) {
        TokenStream tokens = parser.getTokenStream();
        String prototxt = getPrototxt(tokens, ctx.getStart().getTokenIndex(), ctx.getStop().getTokenIndex());

        if (currentLayer.getTops().size() == 1) {
            currentLayer.addAttr("top", currentLayer.getTops().get(0));
        }

        if (currentLayer.getBottoms().size() == 1) {
            currentLayer.addAttr("bottom", currentLayer.getBottoms().get(0));
        }

        currentLayer.setPrototxt(prototxt);
        mlModel.addLayer(currentLayer);
    }

    private String getPrototxt(TokenStream stream, int start, int end) {
        StringBuilder prototxt = new StringBuilder();
        for (int i = start; i <= end; i++) {
            Token token = stream.get(i);
            prototxt.append(token.getText());
        }
        String strPrototxt = prototxt.toString();
        return strPrototxt.replaceAll(" +num_examples:.*\\s", "");
    }

    @Override
    public void enterPair(CaffePrototxtParser.PairContext ctx) {
        String key = ctx.getStart().getText();
        keys.push(key);
    }

    @Override
    public void exitPair(CaffePrototxtParser.PairContext ctx) {

        if (getCurrentKey().equals("param")) {
            currentLayer.getParams().add(currentParams);
            currentParams = new HashMap<>();
        }

        keys.pop();
    }

    @Override
    public void exitValueLeaf(CaffePrototxtParser.ValueLeafContext ctx) {
        String value = ctx.getText();
        value = parserHelper.removeQuotes(value);
        processKeyValue(getCurrentKey(), value);
    }

    protected void processKeyValue(String key, String value) {
        switch (key) {
            case "name":
                currentLayer.setName(value);
                break;
            case "top":
                currentLayer.addTop(value);
                return;
            case "bottom":
                currentLayer.addBottom(value);
                return;
        }

        if (key.toLowerCase().startsWith("param.")) {
            currentParams.put(key, value);
        }

        currentLayer.addAttr(key, value);
    }

    private String getCurrentKey() {
        StringBuilder sb = new StringBuilder();
        for (String s : keys) {
            sb.append(s + ".");
        }
        return sb.substring(0, sb.length() - 1).toString();
    }
}
