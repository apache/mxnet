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

import java.io.IOException;
import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Model;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Predictor;
import org.apache.mxnet.integration.tests.jna.JnaUtilTest;
import org.apache.mxnet.integration.util.Assertions;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.repository.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

public class ModelTest {
    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void modelLoadAndPredictTest() {
        try (MxResource base = BaseMxResource.getSystemMxResource()) {
            Model model = Model.loadModel(Item.MLP);
            //            Model model = Model.loadModel("test",
            // Paths.get("/xxx/xxx/mxnet.java_package/cache/repo/test-models/mlp.tar.gz/mlp/"));
            Predictor<NDList, NDList> predictor = model.newPredictor();
            NDArray input = NDArray.create(base, new Shape(1, 28, 28)).ones();
            NDList inputs = new NDList();
            inputs.add(input);
            NDList result = predictor.predict(inputs);
            NDArray expected =
                    NDArray.create(
                            base,
                            new float[] {
                                4.93476f,
                                -0.76084447f,
                                0.37713608f,
                                0.6605506f,
                                -1.3485785f,
                                -0.8736369f,
                                0.018061712f,
                                -1.3274033f,
                                1.0609543f,
                                0.24042489f
                            },
                            new Shape(1, 10));
            Assertions.assertAlmostEquals(result.get(0), expected);
            logger.info("Trigger ci~");
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
    }
}
