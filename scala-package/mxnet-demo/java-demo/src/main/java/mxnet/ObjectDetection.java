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
package mxnet;
import org.apache.mxnet.infer.javaapi.ObjectDetectorOutput;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.infer.javaapi.ObjectDetector;
import org.apache.commons.io.FileUtils;
import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ObjectDetection {
    private static String modelPath;
    private static String imagePath;

    private static void downloadUrl(String url, String filePath) {
        File tmpFile = new File(filePath);
        if (!tmpFile.exists()) {
            try {
                FileUtils.copyURLToFile(new URL(url), tmpFile);
            } catch (Exception exception) {
                System.err.println(exception);
            }
        }
    }

    public static void downloadModelImage() {
        String tempDirPath = System.getProperty("java.io.tmpdir");
        System.out.println("tempDirPath: %s".format(tempDirPath));
        imagePath = tempDirPath + "/inputImages/resnetssd/dog-ssd.jpg";
        String imgURL = "https://s3.amazonaws.com/model-server/inputs/dog-ssd.jpg";
        downloadUrl(imgURL, imagePath);
        modelPath = tempDirPath + "/resnetssd/resnet50_ssd_model";
        System.out.println("Download model files, this can take a while...");
        String modelURL = "https://s3.amazonaws.com/model-server/models/resnet50_ssd/";
        downloadUrl(modelURL + "resnet50_ssd_model-symbol.json",
                tempDirPath + "/resnetssd/resnet50_ssd_model-symbol.json");
        downloadUrl(modelURL + "resnet50_ssd_model-0000.params",
                tempDirPath + "/resnetssd/resnet50_ssd_model-0000.params");
        downloadUrl(modelURL + "synset.txt",
                tempDirPath + "/resnetssd/synset.txt");
    }

    static List<List<ObjectDetectorOutput>>
    runObjectDetectionSingle(String modelPathPrefix, String inputImagePath, List<Context> context) {
        Shape inputShape = new Shape(new int[] {1, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        return objDet.imageObjectDetect(ObjectDetector.loadImageFromFile(inputImagePath), 3);
    }

    public static void main(String[] args) {
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        downloadModelImage();
        Shape inputShape = new Shape(new int[] {1, 3, 512, 512});
        Shape outputShape = new Shape(new int[] {1, 6132, 6});
        int width = inputShape.get(2);
        int height = inputShape.get(3);
        List<List<ObjectDetectorOutput>> output
                = runObjectDetectionSingle(modelPath, imagePath, context);
        String outputStr = "\n";
        for (List<ObjectDetectorOutput> ele : output) {
            for (ObjectDetectorOutput i : ele) {
                outputStr += "Class: " + i.getClassName() + "\n";
                outputStr += "Probabilties: " + i.getProbability() + "\n";

                List<Float> coord = Arrays.asList(i.getXMin() * width,
                        i.getXMax() * height, i.getYMin() * width, i.getYMax() * height);
                StringBuilder sb = new StringBuilder();
                for (float c: coord) {
                    sb.append(", ").append(c);
                }
                outputStr += "Coord:" + sb.substring(2)+ "\n";
            }
        }
        System.out.println(outputStr);
    }
}