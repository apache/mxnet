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

import org.apache.mxnet.NativeResourceRef;
import org.apache.mxnet.ResourceScope;
import org.junit.Test;

import java.util.*;
import java.util.concurrent.Callable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ResourceScopeTestSuite {

    /**
     * This is a placeholder class to test out whether NDArray References get collected or not when using
     * try-with-resources in Java.
     *
     */
    class TestNDArray  {
        NDArray selfArray;

        public TestNDArray(Context context, int[] shape) {
            this.selfArray = NDArray.ones(context, shape);
        }

        public boolean verifyIsDisposed() {
            return this.selfArray.nd().isDisposed();
        }

        public NativeResourceRef getNDArrayReference() {
            return this.selfArray.nd().ref();
        }
    }

    @Test
    public void testNDArrayAutoRelease() {
        TestNDArray test = null;

        try (ResourceScope scope = new ResourceScope()) {
            test = new TestNDArray(Context.cpu(), new int[]{100, 100});
        }

        assertTrue(test.verifyIsDisposed());
    }

    @Test
    public void testObjectReleaseFromList() {
        List<TestNDArray> list = new ArrayList<>();

        try (ResourceScope scope = new ResourceScope()) {
            for (int i = 0;i < 10; i++) {
                list.add(new TestNDArray(Context.cpu(), new int[] {100, 100}));
            }
        }

        assertEquals(list.size() , 10);
        for (TestNDArray item : list) {
            assertTrue(item.verifyIsDisposed());
        }
    }

    @Test
    public void testObjectReleaseFromMap() {
        Map<String, TestNDArray> stringToNDArrayMap = new HashMap<>();

        try (ResourceScope scope = new ResourceScope()) {
            for (int i = 0;i < 10; i++) {
                stringToNDArrayMap.put(String.valueOf(i),new TestNDArray(Context.cpu(), new int[] {i, i}));
            }
        }

        assertEquals(stringToNDArrayMap.size(), 10);
        for (Map.Entry<String, TestNDArray> entry : stringToNDArrayMap.entrySet()) {
            assertTrue(entry.getValue().verifyIsDisposed());
        }

        Map<TestNDArray, String> ndArrayToStringMap = new HashMap<>();

        try (ResourceScope scope = new ResourceScope()) {
            for (int i = 0;i < 10; i++) {
                ndArrayToStringMap.put(new TestNDArray(Context.cpu(), new int[] {i, i}), String.valueOf(i));
            }
        }

        assertEquals(ndArrayToStringMap.size(), 10);
        for (Map.Entry<TestNDArray, String> entry : ndArrayToStringMap.entrySet()) {
            assertTrue(entry.getKey().verifyIsDisposed());
        }

    }
}
