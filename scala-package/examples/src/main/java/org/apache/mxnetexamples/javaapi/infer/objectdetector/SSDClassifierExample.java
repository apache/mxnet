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

package org.apache.mxnetexamples.javaapi.infer.objectdetector;

import org.apache.mxnet.infer.javaapi.ObjectDetectorOutput;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.infer.javaapi.ObjectDetector;

// scalastyle:off
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
// scalastyle:on

import java.util.*;

import java.io.File;

public class SSDClassifierExample {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "/model/ssd_resnet50_512";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = "/images/dog.jpg";
    @Option(name = "--input-dir", usage = "the input batch of images directory")
    private String inputImageDir = "/images/";
    
    final static Logger logger = LoggerFactory.getLogger(SSDClassifierExample.class);
    
    static List<List<ObjectDetectorOutput>>
    runObjectDetectionSingle(String modelPathPrefix, String inputImagePath, List<Context> context) {
        Shape inputShape = new Shape(new int[]{1, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
        ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        return objDet.imageObjectDetect(img, 3);
    }
    
    static List<List<List<ObjectDetectorOutput>>>
    runObjectDetectionBatch(String modelPathPrefix, String inputImageDir, List<Context> context) {
        Shape inputShape = new Shape(new int[]{1, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        
        // Loading batch of images from the directory path
        List<List<String>> batchFiles = generateBatches(inputImageDir, 20);
        List<List<List<ObjectDetectorOutput>>> outputList
                = new ArrayList<List<List<ObjectDetectorOutput>>>();
        
        for (List<String> batchFile : batchFiles) {
            List<BufferedImage> imgList = ObjectDetector.loadInputBatch(batchFile);
            // Running inference on batch of images loaded in previous step
            List<List<ObjectDetectorOutput>> tmp
                    = objDet.imageBatchObjectDetect(imgList, 5);
            outputList.add(tmp);
        }
        return outputList;
    }
    
    static List<List<String>> generateBatches(String inputImageDirPath, int batchSize) {
        File dir = new File(inputImageDirPath);
        
        List<List<String>> output = new ArrayList<List<String>>();
        List<String> batch = new ArrayList<String>();
        for (File imgFile : dir.listFiles()) {
            batch.add(imgFile.getPath());
            if (batch.size() == batchSize) {
                output.add(batch);
                batch = new ArrayList<String>();
            }
        }
        if (batch.size() > 0) {
            output.add(batch);
        }
        return output;
    }
    
    public static void main(String[] args) {
        SSDClassifierExample inst = new SSDClassifierExample();
        CmdLineParser parser = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }
        
        String mdprefixDir = inst.modelPathPrefix;
        String imgPath = inst.inputImagePath;
        String imgDir = inst.inputImageDir;
        
        if (!checkExist(Arrays.asList(mdprefixDir + "-symbol.json", imgDir, imgPath))) {
            logger.error("Model or input image path does not exist");
            System.exit(1);
        }
        
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        
        try {
            Shape inputShape = new Shape(new int[]{1, 3, 512, 512});
            Shape outputShape = new Shape(new int[]{1, 6132, 6});

            StringBuilder outputStr = new StringBuilder().append("\n");
            
            List<List<ObjectDetectorOutput>> output
                    = runObjectDetectionSingle(mdprefixDir, imgPath, context);

            // Creating Bounding box material
            BufferedImage buf = ImageIO.read(new File(imgPath));
            int width = buf.getWidth();
            int height = buf.getHeight();
            List<Map<String, Integer>> boxes = new ArrayList<>();
            List<String> names = new ArrayList<>();
            for (List<ObjectDetectorOutput> ele : output) {
                for (ObjectDetectorOutput i : ele) {
                    outputStr.append("Class: " + i.getClassName() + "\n");
                    outputStr.append("Probabilties: " + i.getProbability() + "\n");
                    names.add(i.getClassName());
                    Map<String, Integer> map = new HashMap<>();
                    float xmin = i.getXMin() * width;
                    float xmax = i.getXMax() * width;
                    float ymin = i.getYMin() * height;
                    float ymax = i.getYMax() * height;
                    List<Float> coord = Arrays.asList(xmin, xmax, ymin, ymax);
                    map.put("xmin", (int) xmin);
                    map.put("xmax", (int) xmax);
                    map.put("ymin", (int) ymin);
                    map.put("ymax", (int) ymax);
                    boxes.add(map);
                    StringBuilder sb = new StringBuilder();
                    for (float c : coord) {
                        sb.append(", ").append(c);
                    }
                    outputStr.append("Coord:" + sb.substring(2) + "\n");
                }
            }
            logger.info(outputStr.toString());

            // Covert to image
            Image.drawBoundingBox(buf, boxes, names);
            File outputFile = new File("boundingImage.png");
            ImageIO.write(buf, "png", outputFile);

            List<List<List<ObjectDetectorOutput>>> outputList =
                    runObjectDetectionBatch(mdprefixDir, imgDir, context);
            
            outputStr = new StringBuilder().append("\n");
            int index = 0;
            for (List<List<ObjectDetectorOutput>> i : outputList) {
                for (List<ObjectDetectorOutput> j : i) {
                    outputStr.append("*** Image " + (index + 1) + "***" + "\n");
                    for (ObjectDetectorOutput k : j) {
                        outputStr.append("Class: " + k.getClassName() + "\n");
                        outputStr.append("Probabilties: " + k.getProbability() + "\n");
                        List<Float> coord = Arrays.asList(k.getXMin() * width,
                                k.getXMax() * height, k.getYMin() * width, k.getYMax() * height);
                        
                        StringBuilder sb = new StringBuilder();
                        for (float c : coord) {
                            sb.append(", ").append(c);
                        }
                        outputStr.append("Coord:" + sb.substring(2) + "\n");
                    }
                    index++;
                }
            }
            logger.info(outputStr.toString());
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }
        System.exit(0);
    }
    
    static Boolean checkExist(List<String> arr) {
        Boolean exist = true;
        for (String item : arr) {
            if (!(new File(item).exists())) {
                logger.error("Cannot find: " + item);
                exist = false;
            }
        }
        return exist;
    }
}
