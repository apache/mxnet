/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.exception.JnaCallException;
import org.apache.mxnet.jna.JnaUtils;
import org.testng.annotations.Test;

import java.nio.file.Paths;

public class SymbolTest {

    @Test
    public void loadAndCloseTest() {
        try (Symbol symbol =
                     Symbol.loadSymbol(BaseMxResource.getSystemMxResource(),
                             Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json"))) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());
        } catch (JnaCallException e) {
            e.printStackTrace();
        }
    }
}
