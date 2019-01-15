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

package org.apache.mxnetexamples.javaapi.infer.predictor;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This Class is a demo to show how users can use Predictor APIs to do
 * Image Classification with all hand-crafted Pre-processing.
 * All helper functions for image pre-processing are
 * currently available in ObjectDetector class.
 */
public class PredictorExample {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "/model/ssd_resnet50_512";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = "/images/dog.jpg";

    final static Logger logger = LoggerFactory.getLogger(PredictorExample.class);
    private static NDArray$ NDArray = NDArray$.MODULE$;

    /**
     * Helper class to print the maximum prediction result
     * @param probabilities The float array of probability
     * @param modelPathPrefix model Path needs to load the synset.txt
     */
    private static String printMaximumClass(float[] probabilities,
                                            String modelPathPrefix) throws IOException {
        String synsetFilePath = modelPathPrefix.substring(0,
                1 + modelPathPrefix.lastIndexOf(File.separator)) + "/synset.txt";
        BufferedReader reader = new BufferedReader(new FileReader(synsetFilePath));
        ArrayList<String> list = new ArrayList<>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i<probabilities.length;i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
    }

    public static void main(String[] args) {
        PredictorExample inst = new PredictorExample();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }
        // Prepare the model
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        List<DataDesc> inputDesc = new ArrayList<>();
        Shape inputShape = new Shape(new int[]{1, 3, 224, 224});
        inputDesc.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        Predictor predictor = new Predictor(inst.modelPathPrefix, inputDesc, context,0);
        // Prepare data
        NDArray img = Image.imRead(inst.inputImagePath, 1, true);
        img = Image.imResize(img, 224, 224, null);
        // predict
        float[][] result = predictor.predict(new float[][]{img.toArray()});
        try {
            System.out.println("Predict with Float input");
            System.out.println(printMaximumClass(result[0], inst.modelPathPrefix));
        } catch (IOException e) {
            System.err.println(e);
        }
        // predict with NDArray
        NDArray nd = img;
        nd = NDArray.transpose(nd, new Shape(new int[]{2, 0, 1}), null)[0];
        nd = NDArray.expand_dims(nd, 0, null)[0];
        nd = nd.asType(DType.Float32());
        List<NDArray> ndList = new ArrayList<>();
        ndList.add(nd);
        List<NDArray> ndResult = predictor.predictWithNDArray(ndList);
        try {
            System.out.println("Predict with NDArray");
            System.out.println(printMaximumClass(ndResult.get(0).toArray(), inst.modelPathPrefix));
        } catch (IOException e) {
            System.err.println(e);
        }
    }

}
