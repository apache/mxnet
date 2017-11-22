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
 * \file Solver.java
 * \brief Model for the Caffe solver prototxt
 */

package io.mxnet.caffetranslator;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solver {

    private boolean parseDone;
    private Map<String, List<String>> properties;
    private String solverPath;

    public Solver(String solverPath) {
        this.solverPath = solverPath;
        properties = new HashMap<>();
    }

    public void parsePrototxt() {
        CharStream cs = null;
        try {
            FileInputStream fis = new FileInputStream(new File(solverPath));
            cs = CharStreams.fromStream(fis, StandardCharsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }

        CaffePrototxtLexer lexer = new CaffePrototxtLexer(cs);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CaffePrototxtParser parser = new CaffePrototxtParser(tokens);

        SolverListener solverListener = new SolverListener();
        parser.addParseListener(solverListener);
        parser.solver();

        properties = solverListener.getProperties();

        parseDone = true;
    }

    public String getProperty(String key) {
        List<String> list = getProperties(key);
        if (list == null)
            return null;
        return getProperties(key).get(0);
    }

    public List<String> getProperties(String key) {
        if (!parseDone) {
            parsePrototxt();
        }

        return properties.get(key);
    }

    public String getProperty(String key, String defaultValue) {
        String value = getProperty(key);
        if (value == null)
            return defaultValue;
        else
            return value;
    }
}
