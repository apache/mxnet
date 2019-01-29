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

import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.*;

public class PredictorTest {

    Predictor mockPredictor;

    @Before
    public void setUp() {
        mockPredictor = Mockito.mock(Predictor.class);
    }

    @Test
    public void testPredictWithFloatArray() {

        float tmp[][] = new float[1][224];
        for (int x = 0; x < 1; x++) {
            for (int y = 0; y < 224; y++)
                tmp[x][y] = (int) (Math.random() * 10);
        }

        float [][] expectedResult = new float[][] {{1f, 2f}};
        Mockito.when(mockPredictor.predict(tmp)).thenReturn(expectedResult);
        float[][] actualResult = mockPredictor.predict(tmp);

        Mockito.verify(mockPredictor, Mockito.times(1)).predict(tmp);
        Assert.assertArrayEquals(expectedResult, actualResult);
    }

    @Test
    public void testPredictWithNDArray() {

        float[] tmpArr = new float[224];
            for (int y = 0; y < 224; y++)
                tmpArr[y] = (int) (Math.random() * 10);

        NDArray arr = new org.apache.mxnet.javaapi.NDArray(tmpArr, new Shape(new int[] {1, 1, 1, 224}), new Context("cpu", 0));

        List<NDArray> inputList = new ArrayList<>();
        inputList.add(arr);

        NDArray expected = new NDArray(tmpArr, new Shape(new int[] {1, 1, 1, 224}), new Context("cpu", 0));
        List<NDArray> expectedResult = new ArrayList<>();
        expectedResult.add(expected);

        Mockito.when(mockPredictor.predictWithNDArray(inputList)).thenReturn(expectedResult);

        List<NDArray> actualOutput = mockPredictor.predictWithNDArray(inputList);

        Mockito.verify(mockPredictor, Mockito.times(1)).predictWithNDArray(inputList);

        Assert.assertEquals(expectedResult, actualOutput);
    }

    @Test
    public void testPredictWithIterablesNDArray() {

        float[] tmpArr = new float[224];
        for (int y = 0; y < 224; y++)
            tmpArr[y] = (int) (Math.random() * 10);

        NDArray arr = new org.apache.mxnet.javaapi.NDArray(tmpArr, new Shape(new int[] {1, 1, 1, 224}), new Context("cpu", 0));

        Set<NDArray> inputSet = new HashSet<>();
        inputSet.add(arr);

        NDArray expected = new NDArray(tmpArr, new Shape(new int[] {1, 1, 1, 224}), new Context("cpu", 0));
        List<NDArray> expectedResult = new ArrayList<>();
        expectedResult.add(expected);

        Mockito.when(mockPredictor.predictWithNDArray(inputSet)).thenReturn(expectedResult);

        List<NDArray> actualOutput = mockPredictor.predictWithNDArray(inputSet);

        Mockito.verify(mockPredictor, Mockito.times(1)).predictWithNDArray(inputSet);

        Assert.assertEquals(expectedResult, actualOutput);
    }

    @Test
    public void testPredictWithListOfFloatsAsInput() {
        List<List<Float>> input = new ArrayList<>();

        input.add(Arrays.asList(new Float[] {1f, 2f}));

        List<List<Float>> expectedOutput = new ArrayList<>(input);

        Mockito.when(mockPredictor.predict(input)).thenReturn(expectedOutput);

        List<List<Float>> actualOutput = mockPredictor.predict(input);

        Mockito.verify(mockPredictor, Mockito.times(1)).predict(input);

        Assert.assertEquals(expectedOutput, actualOutput);

    }
}