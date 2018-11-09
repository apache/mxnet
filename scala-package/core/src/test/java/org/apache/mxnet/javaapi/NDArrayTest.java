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

package org.apache.mxnet.javaapi;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.javaapi.NDArrayBase.*;

import static org.junit.Assert.assertTrue;

public class NDArrayTest {
    @Test
    public void testCreateNDArray() {
        NDArray nd = new NDArray(new float[]{1.0f, 2.0f, 3.0f},
                new Shape(new int[]{1, 3}),
                new Context("cpu", 0));
        int[] arr = new int[]{1, 3};
        assertTrue(Arrays.equals(nd.shape().toArray(), arr));
        assertTrue(nd.at(0).at(0).toArray()[0] == 1.0f);
        List<Float> list = Arrays.asList(1.0f, 2.0f, 3.0f);
        // Second way creating NDArray
        nd = NDArray.array(list,
                new Shape(new int[]{1, 3}),
                new Context("cpu", 0));
        assertTrue(Arrays.equals(nd.shape().toArray(), arr));
    }

    @Test
    public void testZeroOneEmpty(){
        NDArray ones = NDArray.ones(new Context("cpu", 0), new int[]{100, 100});
        NDArray zeros = NDArray.zeros(new Context("cpu", 0), new int[]{100, 100});
        NDArray empty = NDArray.empty(new Context("cpu", 0), new int[]{100, 100});
        int[] arr = new int[]{100, 100};
        assertTrue(Arrays.equals(ones.shape().toArray(), arr));
        assertTrue(Arrays.equals(zeros.shape().toArray(), arr));
        assertTrue(Arrays.equals(empty.shape().toArray(), arr));
    }

    @Test
    public void testComparison(){
        NDArray nd = new NDArray(new float[]{1.0f, 2.0f, 3.0f}, new Shape(new int[]{3}), new Context("cpu", 0));
        NDArray nd2 = new NDArray(new float[]{3.0f, 4.0f, 5.0f}, new Shape(new int[]{3}), new Context("cpu", 0));
        nd = nd.add(nd2);
        float[] greater = new float[]{1, 1, 1};
        assertTrue(Arrays.equals(nd.greater(nd2).toArray(), greater));
        nd = nd.subtract(nd2);
        nd = nd.subtract(nd2);
        float[] lesser = new float[]{0, 0, 0};
        assertTrue(Arrays.equals(nd.greater(nd2).toArray(), lesser));
    }

    @Test
    public void testGenerated(){
        NDArray$ NDArray = NDArray$.MODULE$;
        float[] arr = new float[]{1.0f, 2.0f, 3.0f};
        NDArray nd = new NDArray(arr, new Shape(new int[]{3}), new Context("cpu", 0));
        float result = NDArray.norm(NDArray.new normParam(nd))[0].toArray()[0];
        float cal = 0.0f;
        for (float ele : arr) {
            cal += ele * ele;
        }
        cal = (float) Math.sqrt(cal);
        assertTrue(Math.abs(result - cal) < 1e-5);
        NDArray dotResult = new NDArray(new float[]{0}, new Shape(new int[]{1}), new Context("cpu", 0));
        NDArray.dot(NDArray.new dotParam(nd, nd).setOut(dotResult));
        assertTrue(Arrays.equals(dotResult.toArray(), new float[]{14.0f}));
    }
}
