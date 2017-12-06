/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Launcher.java
 * \brief Parses command line and invokes Converter
 */

package io.mxnet.caffetranslator;


import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Launcher {

    private String trainingPrototextPath, solverPrototextPath;
    private String paramsFilePath;
    private File outFile;

    protected final String TRAINING_PROTOTXT = "training-prototxt";
    protected final String SOLVER_PROTOTXT = "solver";
    protected final String CUSTOM_DATA_LAYERS = "custom-data-layers";
    protected final String OUTPUT_FILE = "output-file";
    protected final String PARAMS_FILE = "params-file";
    protected final String GRAPH_FILE = "graph-file";


    public static void main(String[] args) {
        Launcher launcher = new Launcher();
        launcher.run(args);
    }

    public void run(String[] args) {
        parseCommandLine(args);

        Converter converter = new Converter(trainingPrototextPath, solverPrototextPath);
        if (paramsFilePath != null) {
            converter.setParamsFilePath(paramsFilePath);
        }
        String code = converter.generateMXNetCode();

        writeToOutFile(code);
        System.out.println("Translated code saved in " + outFile.getAbsolutePath());
    }

    private void writeToOutFile(String code) {
        PrintWriter out;
        try {
            out = new PrintWriter(outFile);
        } catch (FileNotFoundException e) {
            System.err.println(String.format("Unable to open %s for writing", outFile.getAbsoluteFile()));
            return;
        }

        out.print(code);
        out.flush();
    }

    public void parseCommandLine(String[] args) {
        CommandLineParser clParser = new DefaultParser();

        Options options = new Options();

        Option prototxtOption = Option.builder("t")
                .longOpt(TRAINING_PROTOTXT)
                .hasArg()
                .desc("training/validation prototxt")
                .build();
        options.addOption(prototxtOption);

        Option solverOption = Option.builder("s")
                .longOpt(SOLVER_PROTOTXT)
                .hasArg()
                .desc("solver prototxt")
                .build();
        options.addOption(solverOption);

        Option dataLayerOpt = Option.builder("c")
                .longOpt(CUSTOM_DATA_LAYERS)
                .hasArg()
                .desc("Comma separated custom data layers")
                .build();
        options.addOption(dataLayerOpt);

        Option outfileOpt = Option.builder("o")
                .longOpt(OUTPUT_FILE)
                .hasArg()
                .desc("Output file")
                .build();
        options.addOption(outfileOpt);

        Option paramsFileOpt = Option.builder("p")
                .longOpt(PARAMS_FILE)
                .hasArg()
                .desc("Params file")
                .build();
        options.addOption(paramsFileOpt);

        Option graphFileOpt = Option.builder("g")
                .longOpt(GRAPH_FILE)
                .hasArg()
                .desc("Image file to visualize computation graph")
                .build();
        options.addOption(graphFileOpt);

        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            System.out.println("Exception parsing commandline:" + e.getMessage());
            System.exit(1);
        }

        if ((trainingPrototextPath = getOption(line, TRAINING_PROTOTXT)) == null) {
            bail("Command line argument " + TRAINING_PROTOTXT + " missing");
        }

        if ((solverPrototextPath = getOption(line, SOLVER_PROTOTXT)) == null) {
            bail("Command line argument " + SOLVER_PROTOTXT + " missing");
        }

        String strOutFile = getOption(line, OUTPUT_FILE);
        if (strOutFile == null) {
            bail("Command line argument " + OUTPUT_FILE + " missing");
        }
        outFile = new File(strOutFile);

        paramsFilePath = getOption(line, PARAMS_FILE);

        String dataLayers;
        Config config = Config.getInstance();
        if ((dataLayers = getOption(line, CUSTOM_DATA_LAYERS)) != null) {
            for (String name : dataLayers.split(",")) {
                name = name.trim();
                config.addCustomDataLayer(name);
            }
        }

    }

    private String getOption(CommandLine line, String argName) {
        if (line.hasOption(argName)) {
            return line.getOptionValue(argName);
        } else {
            return null;
        }
    }

    private void bail(String reason) {
        System.err.println(reason);
        System.exit(1);
    }
}
