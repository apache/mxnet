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
 * \file BaseGenerator.java
 * \brief Base class for all source generators
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenerationHelper;
import io.mxnet.caffetranslator.SymbolGenerator;
import org.stringtemplate.v4.ST;

import java.util.List;

public abstract class BaseGenerator implements SymbolGenerator {

    protected final GenerationHelper gh;

    public BaseGenerator() {
        gh = new GenerationHelper();
    }

    protected ST getTemplate(String name) {
        return gh.getTemplate(name);
    }

    protected String generateVar(String varName, String symName, String lr_mult, String wd_mult, String init, List<Integer> shape) {
        ST st = getTemplate("var");
        st.add("var", varName);
        st.add("name", symName);

        st.add("lr_mult", (lr_mult == null) ? "None" : lr_mult);
        st.add("wd_mult", (wd_mult == null) ? "None" : wd_mult);
        st.add("init", (init == null) ? "None" : init);
        if (shape != null) {
            st.add("shape", shape);
        }

        return st.render();
    }

}
