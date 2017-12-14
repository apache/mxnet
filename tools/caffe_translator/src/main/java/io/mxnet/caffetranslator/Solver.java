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

import lombok.Getter;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solver {

    private final String solverPath;
    private boolean parseDone;
    private Map<String, List<String>> properties;
    /**
     * Fields corresponding to keys that can be present in the solver prototxt. 'setFields' sets these
     * using reflection after parsing the solver prototxt. A solver object is passed to string templates
     * and the templates read these fields.
     */
    @Getter
    private String base_lr, momentum, weight_decay, lr_policy, gamma, stepsize, stepvalue, max_iter,
            solver_mode, snapshot, snapshot_prefix, test_iter, test_interval, display, type, delta,
            momentum2, rms_decay, solver_type;

    public Solver(String solverPath) {
        this.solverPath = solverPath;
        properties = new HashMap<>();
    }

    public boolean parsePrototxt() {
        CharStream cs = null;
        try {
            FileInputStream fis = new FileInputStream(new File(solverPath));
            cs = CharStreams.fromStream(fis, StandardCharsets.UTF_8);
        } catch (IOException e) {
            System.err.println("Unable to read prototxt " + solverPath);
            return false;
        }

        CaffePrototxtLexer lexer = new CaffePrototxtLexer(cs);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CaffePrototxtParser parser = new CaffePrototxtParser(tokens);

        SolverListener solverListener = new SolverListener();
        parser.addParseListener(solverListener);
        parser.solver();

        properties = solverListener.getProperties();

        setFields(properties);

        parseDone = true;
        return true;
    }

    private void setFields(Map<String, List<String>> properties) {
        Class<?> cls = getClass();

        for (Map.Entry<String, List<String>> entry : properties.entrySet()) {
            String key = entry.getKey();
            try {
                Field field = cls.getDeclaredField(key);
                field.set(this, entry.getValue().get(0));
            } catch (NoSuchFieldException e) {
                // Just ignore
            } catch (IllegalAccessException e) {
                /**
                 * This shouldn't happen. If it does happen because we overlooked something, print
                 * it in the console so we can investigate it.
                 */
                e.printStackTrace();
            }
        }

        setDefaults();
    }

    private void setDefaults() {
        if (type == null) {
            type = "SGD";
        }
        if (delta == null) {
            delta = "1e-8";
        }
        if (momentum2 == null) {
            momentum2 = "0.999";
        }
        if (rms_decay == null) {
            rms_decay = "0.99";
        }
    }

    public String getProperty(String key) {
        List<String> list = getProperties(key);
        if (list == null) {
            return null;
        }
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
        if (value == null) {
            return defaultValue;
        } else {
            return value;
        }
    }
}
