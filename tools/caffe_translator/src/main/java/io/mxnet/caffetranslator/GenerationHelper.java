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
 * \file GenerationHelper.java
 * \brief Helper class used by generators
 */

package io.mxnet.caffetranslator;

import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STErrorListener;
import org.stringtemplate.v4.STGroup;
import org.stringtemplate.v4.STGroupFile;
import org.stringtemplate.v4.STRawGroupDir;
import org.stringtemplate.v4.misc.STMessage;

import java.util.ArrayList;
import java.util.List;

public class GenerationHelper {

    protected final STGroup stGroupDir;

    protected final STGroup stGroupFile;

    private class SuppressSTErrorsListener implements STErrorListener {

        @Override
        public void compileTimeError(STMessage msg) {
            // Do nothing
        }

        @Override
        public void runTimeError(STMessage msg) {
            // Do nothing
        }

        @Override
        public void IOError(STMessage msg) {
            throw new RuntimeException(msg.toString());
        }

        @Override
        public void internalError(STMessage msg) {
            throw new RuntimeException(msg.toString());
        }
    }

    public GenerationHelper() {
        this.stGroupDir = new STRawGroupDir("templates");
        this.stGroupFile = new STGroupFile("templates/symbols.stg");

        SuppressSTErrorsListener errListener = new SuppressSTErrorsListener();
        stGroupDir.setListener(errListener);
        stGroupFile.setListener(errListener);
    }

    public ST getTemplate(String name) {
        ST st = stGroupDir.getInstanceOf(name);
        if (st != null) {
            return st;
        }
        return stGroupFile.getInstanceOf(name);
    }

    public String generateVar(String varName, String symName, String lr_mult, String wd_mult, String init, List<Integer> shape) {
        ST st = getTemplate("var");
        st.add("var", varName);
        st.add("name", symName);

        st.add("lr_mult", lr_mult);
        st.add("wd_mult", wd_mult);
        st.add("init", init);
        st.add("shape", shape);

        return st.render();
    }

    public String getInit(String fillerType, String fillerValue) {
        if (fillerType == null && fillerValue == null) {
            return null;
        }

        if (fillerType == null) {
            fillerType = "constant";
        }

        if (fillerValue == null) {
            fillerValue = "0";
        }

        String initializer;
        switch (fillerType) {
            case "xavier":
                initializer = "mx.initializer.Xavier()";
                break;
            case "gaussian":
                initializer = "mx.initializer.Normal()";
                break;
            case "constant":
                initializer = String.format("mx.initializer.Constant(%s)", fillerValue);
                break;
            case "bilinear":
                initializer = "mx.initializer.Bilinear()";
                break;
            default:
                initializer = "UnknownInitializer";
                System.err.println("Initializer " + fillerType + " not supported");
                break;
        }

        return initializer;
    }

    public String getVarname(String name) {
        StringBuilder sb = new StringBuilder(name);
        for (int i = 0; i < sb.length(); i++) {
            char ch = sb.charAt(i);
            if (Character.isLetter(ch) || Character.isDigit(ch) || ch == '_') {
                // do nothing
            } else {
                sb.replace(i, i + 1, "_");
            }
        }
        return sb.toString();
    }

    public List<String> getVarNames(List<String> names) {
        List<String> list = new ArrayList<>();
        for (String name : names) {
            list.add(getVarname(name));
        }
        return list;
    }

    public void fillNameDataAndVar(ST st, Layer layer) {
        st.add("name", layer.getName());
        st.add("data", getVarname(layer.getBottom()));
        st.add("var", getVarname(layer.getTop()));
    }

    public void simpleFillTemplate(ST st, String name, Layer layer, String key, String defaultValue, String... altKeys) {
        String value = layer.getAttr(key);

        if (value == null) {
            for (String altKey : altKeys) {
                value = layer.getAttr(altKey);
                if (value != null) {
                    break;
                }
            }
        }

        if (value == null && defaultValue != null) {
            value = defaultValue;
        }

        if (value == null) {
            System.err.println(String.format("Layer %s does not contain attribute %s or alternates",
                    layer.getName(), key));
            value = "???";
        }

        st.add(name, value);
    }

    public GeneratorOutput makeGeneratorOutput(String code, int numLayersTranslated) {
        return new GeneratorOutput(code, numLayersTranslated);
    }

    public String initializeParam(String varname, int childIndex, String initializer) {
        StringBuilder out = new StringBuilder();
        out.append(String.format("param_initializer.add_param(%s.get_children()[%d].name, %s)",
                varname, childIndex, initializer));
        out.append(System.lineSeparator());
        return out.toString();
    }
}
