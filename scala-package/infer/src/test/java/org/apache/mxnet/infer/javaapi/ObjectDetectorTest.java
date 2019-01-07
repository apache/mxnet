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

import org.apache.mxnet.Layout;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class ObjectDetectorTest {

    List<DataDesc> inputDesc;
    BufferedImage inputImage;

    List<List<ObjectDetectorOutput>> result;

    ObjectDetector objectDetector;

    @Before
    public void setUp() {

        inputDesc = new ArrayList<>();
        inputDesc.add(new DataDesc("", new Shape(new int[]{1, 3, 512, 512}), DType.Float32(), Layout.NCHW()));
        inputImage = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        objectDetector = Mockito.mock(ObjectDetector.class);
        result = new ArrayList<>();
        result.add(new ArrayList<ObjectDetectorOutput>());
        result.get(0).add(new ObjectDetectorOutput("simbaa", new float[]{}));
    }

    @Test
    public void testObjectDetectorWithInputImage() {

        Mockito.when(objectDetector.imageObjectDetect(inputImage, 5)).thenReturn(result);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.imageObjectDetect(inputImage, 5);
        Mockito.verify(objectDetector, Mockito.times(1)).imageObjectDetect(inputImage, 5);
        Assert.assertEquals(result, actualResult);
    }


    @Test
    public void testObjectDetectorWithBatchImage() {

        List<BufferedImage> batchImage = new ArrayList<>();
        batchImage.add(inputImage);
        Mockito.when(objectDetector.imageBatchObjectDetect(batchImage, 5)).thenReturn(result);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.imageBatchObjectDetect(batchImage, 5);
        Mockito.verify(objectDetector, Mockito.times(1)).imageBatchObjectDetect(batchImage, 5);
        Assert.assertEquals(result, actualResult);
    }

    @Test
    public void testObjectDetectorWithNDArrayInput() {

        NDArray inputArr = ObjectDetector.bufferedImageToPixels(inputImage, new Shape(new int[] {1, 3, 512, 512}));
        List<NDArray> inputL = new ArrayList<>();
        inputL.add(inputArr);
        Mockito.when(objectDetector.objectDetectWithNDArray(inputL, 5)).thenReturn(result);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.objectDetectWithNDArray(inputL, 5);
        Mockito.verify(objectDetector, Mockito.times(1)).objectDetectWithNDArray(inputL, 5);
        Assert.assertEquals(result, actualResult);
    }
}
