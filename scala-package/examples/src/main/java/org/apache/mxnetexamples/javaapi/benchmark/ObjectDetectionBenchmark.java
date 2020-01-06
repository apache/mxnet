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

package org.apache.mxnetexamples.javaapi.benchmark;

import org.apache.mxnet.infer.javaapi.ObjectDetector;
import org.apache.mxnet.javaapi.*;
import org.kohsuke.args4j.Option;

import java.util.ArrayList;
import java.util.List;

class ObjectDetectionBenchmark extends InferBase {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    public String modelPathPrefix = "/model/ssd_resnet50_512";
    @Option(name = "--input-image", usage = "the input image")
    public String inputImagePath = "/images/dog.jpg";

    private ObjectDetector objDet;
    private NDArray img;
    private NDArray$ NDArray = NDArray$.MODULE$;

    public void preProcessModel(List<Context> context) {
        Shape inputShape = new Shape(new int[] {this.batchSize, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        img = ObjectDetector.bufferedImageToPixels(
                ObjectDetector.reshapeImage(
                    ObjectDetector.loadImageFromFile(inputImagePath), 512, 512
                ),
                new Shape(new int[] {1, 3, 512, 512})
        );
    }

    public void runSingleInference() {
        List<NDArray> nd = new ArrayList<>();
        nd.add(img);
        objDet.objectDetectWithNDArray(nd, 3);
    }

    public void runBatchInference() {
        List<NDArray> nd = new ArrayList<>();
        NDArray[] temp = new NDArray[batchSize];
        for (int i = 0; i < batchSize; i++) temp[i] = img.copy();
        NDArray batched = NDArray.concat(temp, batchSize, 0, null)[0];
        nd.add(batched);
        objDet.objectDetectWithNDArray(nd, 3);
    }
}
