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

import org.apache.mxnet.javaapi.Context;
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JavaBenchmark {

    private static boolean runBatch = false;

    private static void parse(Object inst, String[] args) {
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            System.err.println(e.getMessage() + e);
            parser.printUsage(System.err);
            System.exit(1);
        }
    }

    private static long percentile(int p, long[] seq) {
        Arrays.sort(seq);
        int k = (int) Math.ceil((seq.length - 1) * (p / 100.0));
        return seq[k];
    }

    private static void printStatistics(long[] inferenceTimesRaw, String metricsPrefix)  {
        long[] inferenceTimes = inferenceTimesRaw;
        // remove head and tail
        if (inferenceTimes.length > 2) {
            inferenceTimes = Arrays.copyOfRange(inferenceTimesRaw,
                    1, inferenceTimesRaw.length - 1);
        }
        double p50 = percentile(50, inferenceTimes) / 1.0e6;
        double p99 = percentile(99, inferenceTimes) / 1.0e6;
        double p90 = percentile(90, inferenceTimes) / 1.0e6;
        long sum = 0;
        for (long time: inferenceTimes) sum += time;
        double average = sum / (inferenceTimes.length * 1.0e6);

        System.out.println(
                String.format("\n%s_p99 %fms\n%s_p90 %fms\n%s_p50 %fms\n%s_average %1.2fms",
                        metricsPrefix, p99, metricsPrefix, p90,
                        metricsPrefix, p50, metricsPrefix, average)
        );

    }

    private static List<Context> bindToDevice()  {
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        return context;
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            StringBuilder sb = new StringBuilder();
            sb.append("Please follow the format:");
            sb.append("\n  --model-name <model-name>");
            sb.append("\n  --num-runs <number of runs>");
            sb.append("\n  --batchsize <batch size>");
            System.out.println(sb.toString());
            return;
        }
        String modelName = args[1];
        InferBase model = null;
        switch(modelName) {
            case "ObjectDetection":
                runBatch = true;
                ObjectDetectionBenchmark inst = new ObjectDetectionBenchmark();
                parse(inst, args);
                model = inst;
                break;
            default:
                System.err.println("Model name not found! " + modelName);
                System.exit(1);
        }
        List<Context> context = bindToDevice();
        long[] result = new long[model.numRun];
        model.preProcessModel(context);
        if (runBatch) {
            for (int i =0;i < model.numRun; i++) {
                long currTime = System.nanoTime();
                model.runBatchInference();
                result[i] = System.nanoTime() - currTime;
            }
            System.out.println("Batchsize: " + model.batchSize);
            System.out.println("Num of runs: " + model.numRun);
            printStatistics(result, modelName +"batch_inference");
        }

        model.batchSize = 1;
        model.preProcessModel(context);
        result = new long[model.numRun];
        for (int i = 0; i < model.numRun; i++) {
            long currTime = System.nanoTime();
            model.runSingleInference();
            result[i] = System.nanoTime() - currTime;
        }
        System.out.println("Num of runs: " + model.numRun);
        printStatistics(result, modelName + "single_inference");
    }
}
