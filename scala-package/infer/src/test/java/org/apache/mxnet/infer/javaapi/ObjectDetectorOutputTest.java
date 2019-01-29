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

package org.apache.mxnet.infer.javaapi;

import org.junit.Assert;
import org.junit.Test;

public class ObjectDetectorOutputTest {

    private String predictedClassName = "lion";

    private float delta = 0.00001f;

    @Test
    public void testConstructor() {

        float[] arr = new float[]{0f, 1f, 2f, 3f, 4f};

        ObjectDetectorOutput odOutput = new ObjectDetectorOutput(predictedClassName, arr);

        Assert.assertEquals(odOutput.getClassName(), predictedClassName);
        Assert.assertEquals("Threshold not matching", odOutput.getProbability(), 0f, delta);
        Assert.assertEquals("Threshold not matching", odOutput.getXMin(), 1f, delta);
        Assert.assertEquals("Threshold not matching", odOutput.getXMax(), 2f, delta);
        Assert.assertEquals("Threshold not matching", odOutput.getYMin(), 3f, delta);
        Assert.assertEquals("Threshold not matching", odOutput.getYMax(), 4f, delta);

    }

    @Test (expected = ArrayIndexOutOfBoundsException.class)
    public void testIncompleteArgsConstructor() {

        float[] arr = new float[]{0f, 1f};

        ObjectDetectorOutput odOutput = new ObjectDetectorOutput(predictedClassName, arr);

        Assert.assertEquals(odOutput.getClassName(), predictedClassName);
        Assert.assertEquals("Threshold not matching", odOutput.getProbability(), 0f, delta);
        Assert.assertEquals("Threshold not matching", odOutput.getXMin(), 1f, delta);

        // This is where exception will be thrown
        odOutput.getXMax();
    }
}
