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
package org.apache.mxnet.jnarator;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    private Main() {}

    public static void main(String[] args) {
        Options options = Config.getOptions();
        try {
            DefaultParser cmdParser = new DefaultParser();
            CommandLine cmd = cmdParser.parse(options, args, null, false);
            Config config = new Config(cmd);

            String output = config.getOutput();
            String packageName = config.getPackageName();
            String library = config.getLibrary();
            String[] headerFiles = config.getHeaderFiles();
            String mappingFile = config.getMappingFile();

            Path dir = Paths.get(output);
            Files.createDirectories(dir);

            Properties mapping = new Properties();
            if (mappingFile != null) {
                Path file = Paths.get(mappingFile);
                if (Files.notExists(file)) {
                    logger.error("mapping file does not exists: {}", mappingFile);
                    System.exit(-1); // NOPMD
                }
                try (InputStream in = Files.newInputStream(file)) {
                    mapping.load(in);
                }
            }

            JnaParser jnaParser = new JnaParser();
            Map<String, TypeDefine> typedefMap = jnaParser.getTypedefMap();
            Map<String, List<TypeDefine>> structMap = jnaParser.getStructMap();
            JnaGenerator generator =
                    new JnaGenerator(library, packageName, typedefMap, structMap.keySet(), mapping);
            generator.init(output);

            for (String headerFile : headerFiles) {
                jnaParser.parse(headerFile);
            }

            generator.writeNativeSize();
            generator.writeStructure(structMap);
            generator.writeLibrary(jnaParser.getFunctions(), jnaParser.getEnumMap());
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
            System.exit(-1); // NOPMD
        } catch (Throwable t) {
            logger.error("", t);
            System.exit(-1); // NOPMD
        }
    }

    public static final class Config {

        private String library;
        private String packageName;
        private String output;
        private String[] headerFiles;
        private String mappingFile;

        public Config(CommandLine cmd) {
            library = cmd.getOptionValue("library");
            packageName = cmd.getOptionValue("package");
            output = cmd.getOptionValue("output");
            headerFiles = cmd.getOptionValues("header");
            mappingFile = cmd.getOptionValue("mappingFile");
        }

        public static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("l")
                            .longOpt("library")
                            .hasArg()
                            .required()
                            .argName("LIBRARY")
                            .desc("library name")
                            .build());
            options.addOption(
                    Option.builder("p")
                            .longOpt("package")
                            .required()
                            .hasArg()
                            .argName("PACKAGE")
                            .desc("Java package name")
                            .build());
            options.addOption(
                    Option.builder("o")
                            .longOpt("output")
                            .required()
                            .hasArg()
                            .argName("OUTPUT")
                            .desc("output directory")
                            .build());
            options.addOption(
                    Option.builder("f")
                            .longOpt("header")
                            .required()
                            .hasArgs()
                            .argName("HEADER")
                            .desc("Header files")
                            .build());
            options.addOption(
                    Option.builder("m")
                            .longOpt("mappingFile")
                            .hasArg()
                            .argName("MAPPING_FILE")
                            .desc("Type mappingFile config file")
                            .build());
            return options;
        }

        public String getLibrary() {
            return library;
        }

        public String getPackageName() {
            return packageName;
        }

        public String getOutput() {
            return output;
        }

        public String[] getHeaderFiles() {
            return headerFiles;
        }

        public String getMappingFile() {
            return mappingFile;
        }
    }
}
