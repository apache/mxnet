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

package org.apache.mxnetexamples.javaapi.infer.bert;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * This is an example of using BERT to do the general Question and Answer inference jobs
 * Users can provide a question with a paragraph contains answer to the model and
 * the model will be able to find the best answer from the answer paragraph
 */
public class BertQA {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "/model/static_bert_qa";
    @Option(name = "--model-epoch", usage = "Epoch number of the model")
    private int epoch = 2;
    @Option(name = "--model-vocab", usage = "the vocabulary used in the model")
    private String modelVocab = "/model/vocab.json";
    @Option(name = "--input-question", usage = "the input question")
    private String inputQ = "When did BBC Japan start broadcasting?";
    @Option(name = "--input-answer", usage = "the input answer")
    private String inputA =
        "BBC Japan was a general entertainment Channel.\n" +
                " Which operated between December 2004 and April 2006.\n" +
            "It ceased operations after its Japanese distributor folded.";
    @Option(name = "--seq-length", usage = "the maximum length of the sequence")
    private int seqLength = 384;

    private final static Logger logger = LoggerFactory.getLogger(BertQA.class);
    private static NDArray$ NDArray = NDArray$.MODULE$;

    private static int argmax(float[] prob) {
        int maxIdx = 0;
        for (int i = 0; i < prob.length; i++) {
            if (prob[maxIdx] < prob[i]) maxIdx = i;
        }
        return maxIdx;
    }

    /**
     * Do the post processing on the output, apply softmax to get the probabilities
     * reshape and get the most probable index
     * @param result prediction result
     * @param tokens word tokens
     * @return Answers clipped from the original paragraph
     */
    static List<String> postProcessing(NDArray result, List<String> tokens) {
        NDArray[] output = NDArray.split(
                new splitParam(result, 2).setAxis(2));
        // Get the formatted logits result
        NDArray startLogits = output[0].reshape(new int[]{0, -3});
        NDArray endLogits = output[1].reshape(new int[]{0, -3});
        // Get Probability distribution
        float[] startProb = NDArray.softmax(
                new softmaxParam(startLogits))[0].toArray();
        float[] endProb = NDArray.softmax(
                new softmaxParam(endLogits))[0].toArray();
        int startIdx = argmax(startProb);
        int endIdx = argmax(endProb);
        return tokens.subList(startIdx, endIdx + 1);
    }

    public static void main(String[] args) throws Exception{
        BertQA inst = new BertQA();
        CmdLineParser parser = new CmdLineParser(inst);
        parser.parseArgument(args);
        BertDataParser util = new BertDataParser();
        Context context = Context.cpu();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context = Context.gpu();
        }
        // pre-processing - tokenize sentence
        List<String> tokenQ = util.tokenizer(inst.inputQ.toLowerCase());
        List<String> tokenA = util.tokenizer(inst.inputA.toLowerCase());
        int validLength = tokenQ.size() + tokenA.size();
        logger.info("Valid length: " + validLength);
        // generate token types [0000...1111....0000]
        List<Float> QAEmbedded = new ArrayList<>();
        util.pad(QAEmbedded, 0f, tokenQ.size()).addAll(
                util.pad(new ArrayList<Float>(), 1f, tokenA.size())
        );
        List<Float> tokenTypes = util.pad(QAEmbedded, 0f, inst.seqLength);
        // make BERT pre-processing standard
        tokenQ.add("[SEP]");
        tokenQ.add(0, "[CLS]");
        tokenA.add("[SEP]");
        tokenQ.addAll(tokenA);
        List<String> tokens = util.pad(tokenQ, "[PAD]", inst.seqLength);
        logger.info("Pre-processed tokens: " + Arrays.toString(tokenQ.toArray()));
        // pre-processing - token to index translation
        util.parseJSON(inst.modelVocab);
        List<Integer> indexes = util.token2idx(tokens);
        List<Float> indexesFloat = new ArrayList<>();
        for (int integer : indexes) {
            indexesFloat.add((float) integer);
        }
        // Preparing the input data
        List<NDArray> inputBatch = Arrays.asList(
                new NDArray(indexesFloat,
                        new Shape(new int[]{1, inst.seqLength}), context),
                new NDArray(tokenTypes,
                        new Shape(new int[]{1, inst.seqLength}), context),
                new NDArray(new float[] { validLength },
                        new Shape(new int[]{1}), context)
        );
        // Build the model
        List<Context> contexts = new ArrayList<>();
        contexts.add(context);
        List<DataDesc> inputDescs = Arrays.asList(
                new DataDesc("data0",
                        new Shape(new int[]{1, inst.seqLength}), DType.Float32(), Layout.NT()),
                new DataDesc("data1",
                        new Shape(new int[]{1, inst.seqLength}), DType.Float32(), Layout.NT()),
                new DataDesc("data2",
                        new Shape(new int[]{1}), DType.Float32(), Layout.N())
        );
        Predictor bertQA = new Predictor(inst.modelPathPrefix, inputDescs, contexts, inst.epoch);
        // Start prediction
        NDArray result = bertQA.predictWithNDArray(inputBatch).get(0);
        List<String> answer = postProcessing(result, tokens);
        logger.info("Question: " + inst.inputQ);
        logger.info("Answer paragraph: " + inst.inputA);
        logger.info("Answer: " + Arrays.toString(answer.toArray()));
    }
}
