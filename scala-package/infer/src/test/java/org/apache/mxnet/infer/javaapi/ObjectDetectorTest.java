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

    private List<DataDesc> inputDesc;
    private BufferedImage inputImage;

    private List<List<ObjectDetectorOutput>> expectedResult;

    private ObjectDetector objectDetector;

    private int batchSize = 1;

    private int channels = 3;

    private int imageHeight = 512;

    private int imageWidth = 512;

    private String dataName = "data";

    private int topK = 5;

    private String predictedClassName = "lion"; // Random string

    private Shape getTestShape() {

        return new Shape(new int[] {batchSize, channels, imageHeight, imageWidth});
    }

    @Before
    public void setUp() {

        inputDesc = new ArrayList<>();
        inputDesc.add(new DataDesc(dataName, getTestShape(), DType.Float32(), Layout.NCHW()));
        inputImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
        objectDetector = Mockito.mock(ObjectDetector.class);
        expectedResult = new ArrayList<>();
        expectedResult.add(new ArrayList<ObjectDetectorOutput>());
        expectedResult.get(0).add(new ObjectDetectorOutput(predictedClassName, new float[]{}));
    }

    @Test
    public void testObjectDetectorWithInputImage() {

        Mockito.when(objectDetector.imageObjectDetect(inputImage, topK)).thenReturn(expectedResult);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.imageObjectDetect(inputImage, topK);
        Mockito.verify(objectDetector, Mockito.times(1)).imageObjectDetect(inputImage, topK);
        Assert.assertEquals(expectedResult, actualResult);
    }


    @Test
    public void testObjectDetectorWithBatchImage() {

        List<BufferedImage> batchImage = new ArrayList<>();
        batchImage.add(inputImage);
        Mockito.when(objectDetector.imageBatchObjectDetect(batchImage, topK)).thenReturn(expectedResult);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.imageBatchObjectDetect(batchImage, topK);
        Mockito.verify(objectDetector, Mockito.times(1)).imageBatchObjectDetect(batchImage, topK);
        Assert.assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testObjectDetectorWithNDArrayInput() {

        NDArray inputArr = ObjectDetector.bufferedImageToPixels(inputImage, getTestShape());
        List<NDArray> inputL = new ArrayList<>();
        inputL.add(inputArr);
        Mockito.when(objectDetector.objectDetectWithNDArray(inputL, 5)).thenReturn(expectedResult);
        List<List<ObjectDetectorOutput>> actualResult = objectDetector.objectDetectWithNDArray(inputL, topK);
        Mockito.verify(objectDetector, Mockito.times(1)).objectDetectWithNDArray(inputL, topK);
        Assert.assertEquals(expectedResult, actualResult);
    }
}
