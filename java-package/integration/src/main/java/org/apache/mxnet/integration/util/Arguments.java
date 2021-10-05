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

package org.apache.mxnet.integration.util;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class Arguments {

    private String methodName;
    private String className;
    private String packageName;
    private int duration;
    private int iteration = 1;

    public Arguments(CommandLine cmd) {
        methodName = cmd.getOptionValue("method-name");
        className = cmd.getOptionValue("class-name");
        if (cmd.hasOption("package-name")) {
            packageName = cmd.getOptionValue("package-name");
        } else {
            packageName = "org.apache.mxnet.integration.tests.";
        }

        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        if (cmd.hasOption("iteration")) {
            iteration = Integer.parseInt(cmd.getOptionValue("iteration"));
        }
    }

    public static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("d")
                        .longOpt("duration")
                        .hasArg()
                        .argName("DURATION")
                        .desc("Duration of the test.")
                        .build());
        options.addOption(
                Option.builder("n")
                        .longOpt("iteration")
                        .hasArg()
                        .argName("ITERATION")
                        .desc("Number of iterations in each test.")
                        .build());
        options.addOption(
                Option.builder("p")
                        .longOpt("package-name")
                        .hasArg()
                        .argName("PACKAGE-NAME")
                        .desc("Name of the package to run")
                        .build());
        options.addOption(
                Option.builder("c")
                        .longOpt("class-name")
                        .hasArg()
                        .argName("CLASS-NAME")
                        .desc("Name of the class to run")
                        .build());
        options.addOption(
                Option.builder("m")
                        .longOpt("method-name")
                        .hasArg()
                        .argName("METHOD-NAME")
                        .desc("Name of the method to run")
                        .build());
        return options;
    }

    public int getDuration() {
        return duration;
    }

    public int getIteration() {
        return iteration;
    }

    public String getPackageName() {
        return packageName;
    }

    public String getClassName() {
        return className;
    }

    public String getMethodName() {
        return methodName;
    }
}
