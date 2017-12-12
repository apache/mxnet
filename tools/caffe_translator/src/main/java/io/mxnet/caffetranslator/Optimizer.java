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
 * \file Optimizer.java
 * \brief Generates optimizer from solver prototxt
 */

package io.mxnet.caffetranslator;

import org.stringtemplate.v4.ST;

public class Optimizer {
    private final GenerationHelper gh;
    private final Solver solver;

    public Optimizer(Solver solver) {
        this.gh = new GenerationHelper();
        this.solver = solver;
    }

    public String generateInitCode() {
        ST st = gh.getTemplate("opt_" + solver.getType().toLowerCase());
        if (st == null) {
            System.err.println(String.format("Unknown optimizer type (%s). Using SGD instead.", solver.getType()));
            st = gh.getTemplate("opt_sgd");
        }

        st.add("solver", solver);
        return st.render();
    }
}
