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
 * \file SolverListener.java
 * \brief ANTLR listener that builds the Solver instance as the solver prototxt is parsed
 */

package io.mxnet.caffetranslator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SolverListener extends CaffePrototxtBaseListener {

    private final Map<String, List<String>> properties;
    private final ParserHelper parserHelper;

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

        if (properties.get(key) == null) {
            properties.put(key, new ArrayList<>());
        }

        properties.get(key).add(value);
    }
}
