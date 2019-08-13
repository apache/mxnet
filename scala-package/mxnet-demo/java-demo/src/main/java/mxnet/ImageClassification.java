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

import org.apache.commons.io.FileUtils;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class ImageClassification {
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
        String baseUrl = "https://s3.us-east-2.amazonaws.com/scala-infer-models";
        downloadUrl(baseUrl + "/resnet-18/resnet-18-symbol.json",
                tempDirPath + "/resnet18/resnet-18-symbol.json");
        downloadUrl(baseUrl + "/resnet-18/resnet-18-0000.params",
                tempDirPath + "/resnet18/resnet-18-0000.params");
        downloadUrl(baseUrl + "/resnet-18/synset.txt",
                tempDirPath + "/resnet18/synset.txt");
        downloadUrl("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg",
                tempDirPath + "/inputImages/resnet18/Pug-Cookie.jpg");
        modelPath = tempDirPath + File.separator + "resnet18/resnet-18";
        imagePath = tempDirPath + File.separator +
                "inputImages/resnet18/Pug-Cookie.jpg";
    }

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
        // Download the model and Image
        downloadModelImage();

        // Prepare the model
        List<Context> context = new ArrayList<Context>();
        context.add(Context.cpu());
        List<DataDesc> inputDesc = new ArrayList<>();
        Shape inputShape = new Shape(new int[]{1, 3, 224, 224});
        inputDesc.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        Predictor predictor = new Predictor(modelPath, inputDesc, context,0);

        // Prepare data
        NDArray nd = Image.imRead(imagePath, 1, true);
        nd = Image.imResize(nd, 224, 224, null);
        nd = NDArray.transpose(nd, new Shape(new int[]{2, 0, 1}), null)[0];  // HWC to CHW
        nd = NDArray.expand_dims(nd, 0, null)[0]; // Add N -> NCHW
        nd = nd.asType(DType.Float32()); // Inference with Float32

        // Predict directly
        float[][] result = predictor.predict(new float[][]{nd.toArray()});
        try {
            System.out.println("Predict with Float input");
            System.out.println(printMaximumClass(result[0], modelPath));
        } catch (IOException e) {
            System.err.println(e);
        }

        // predict with NDArray
        List<NDArray> ndList = new ArrayList<>();
        ndList.add(nd);
        List<NDArray> ndResult = predictor.predictWithNDArray(ndList);
        try {
            System.out.println("Predict with NDArray");
            System.out.println(printMaximumClass(ndResult.get(0).toArray(), modelPath));
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
