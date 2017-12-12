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
 * \file Converter.java
 * \brief Convert Caffe prototxt to MXNet Python code
 */

package io.mxnet.caffetranslator;

import io.mxnet.caffetranslator.generators.*;
import lombok.Setter;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroup;
import org.stringtemplate.v4.STRawGroupDir;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Converter {

    private final String trainPrototxt, solverPrototxt;
    private final MLModel mlModel;
    private final STGroup stGroup;
    private final SymbolGeneratorFactory generators;
    private final String NL;
    private final GenerationHelper gh;
    @Setter

    private String paramsFilePath;
    private Solver solver;

    Converter(String trainPrototxt, String solverPrototxt) {
        this.trainPrototxt = trainPrototxt;
        this.solverPrototxt = solverPrototxt;
        this.mlModel = new MLModel();
        this.stGroup = new STRawGroupDir("templates");
        this.generators = SymbolGeneratorFactory.getInstance();
        NL = System.getProperty("line.separator");
        gh = new GenerationHelper();
        addGenerators();
    }

    private void addGenerators() {
        generators.addGenerator("Convolution", new ConvolutionGenerator());
        generators.addGenerator("Deconvolution", new DeconvolutionGenerator());
        generators.addGenerator("Pooling", new PoolingGenerator());
        generators.addGenerator("InnerProduct", new FCGenerator());
        generators.addGenerator("ReLU", new ReluGenerator());
        generators.addGenerator("SoftmaxWithLoss", new SoftmaxOutputGenerator());
        generators.addGenerator("PluginIntLayerGenerator", new PluginIntLayerGenerator());
        generators.addGenerator("CaffePluginLossLayer", new PluginLossGenerator());
        generators.addGenerator("Permute", new PermuteGenerator());
        generators.addGenerator("Concat", new ConcatGenerator());
        generators.addGenerator("BatchNorm", new BatchNormGenerator());
        generators.addGenerator("Power", new PowerGenerator());
        generators.addGenerator("Eltwise", new EltwiseGenerator());
        generators.addGenerator("Flatten", new FlattenGenerator());
        generators.addGenerator("Dropout", new DropoutGenerator());
        generators.addGenerator("Scale", new ScaleGenerator());
    }

    public boolean parseTrainingPrototxt() {

        CharStream cs = null;
        try {
            FileInputStream fis = new FileInputStream(new File(trainPrototxt));
            cs = CharStreams.fromStream(fis, StandardCharsets.UTF_8);
        } catch (IOException e) {
            System.err.println("Unable to read prototxt: " + trainPrototxt);
            return false;
        }

        CaffePrototxtLexer lexer = new CaffePrototxtLexer(cs);

        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CaffePrototxtParser parser = new CaffePrototxtParser(tokens);

        CreateModelListener modelCreator = new CreateModelListener(parser, mlModel);
        parser.addParseListener(modelCreator);
        parser.prototxt();

        return true;
    }

    public boolean parseSolverPrototxt() {
        solver = new Solver(solverPrototxt);
        return solver.parsePrototxt();
    }

    public String generateMXNetCode() {
        if (!parseTrainingPrototxt()) {
            return "";
        }

        if (!parseSolverPrototxt()) {
            return "";
        }

        StringBuilder code = new StringBuilder();

        code.append(generateImports());
        code.append(System.lineSeparator());

        code.append(generateLogger());
        code.append(System.lineSeparator());

        code.append(generateParamInitializer());
        code.append(System.lineSeparator());

        code.append(generateMetricsClasses());
        code.append(System.lineSeparator());

        if (paramsFilePath != null) {
            code.append(generateParamsLoader());
            code.append(System.lineSeparator());
        }

        // Convert data layers
        code.append(generateIterators());

        // Generate variables for data and label
        code.append(generateInputVars());

        // Convert non data layers
        List<Layer> layers = mlModel.getNonDataLayers();

        for (int layerIndex = 0; layerIndex < layers.size(); ) {
            Layer layer = layers.get(layerIndex);
            SymbolGenerator generator = generators.getGenerator(layer.getType());

            // Handle layers for which there is no Generator
            if (generator == null) {
                if (layer.getType().equalsIgnoreCase("Accuracy")) {
                    // We handle accuracy layers at a later stage. Do nothing for now.
                } else if (layer.getType().toLowerCase().endsWith("loss")) {
                    // This is a loss layer we don't have a generator for. Wrap it in CaffeLoss.
                    generator = generators.getGenerator("CaffePluginLossLayer");
                } else {
                    // This is a layer we don't have a generator for. Wrap it in CaffeOp.
                    generator = generators.getGenerator("PluginIntLayerGenerator");
                }
            }

            if (generator != null) { // If we have a generator
                // Generate code
                GeneratorOutput out = generator.generate(layer, mlModel);
                String segment = out.code;
                code.append(segment);
                code.append(NL);

                // Update layerIndex depending on how many layers we ended up translating
                layerIndex += out.numLayersTranslated;
            } else { // If we don't have a generator
                // We've decided to skip this layer. Generate no code. Just increment layerIndex
                // by 1 and move on to the next layer.
                layerIndex++;
            }
        }

        String loss = getLoss(mlModel, code);

        String evalMetric = generateValidationMetrics(mlModel);
        code.append(evalMetric);

        String runner = generateRunner(loss);
        code.append(runner);

        return code.toString();
    }

    private String generateLogger() {
        ST st = gh.getTemplate("logging");
        st.add("name", mlModel.getName());
        return st.render();
    }

    private String generateRunner(String loss) {
        ST st = gh.getTemplate("runner");
        st.add("max_iter", solver.getProperty("max_iter"));
        st.add("stepsize", solver.getProperty("stepsize"));
        st.add("snapshot", solver.getProperty("snapshot"));
        st.add("test_interval", solver.getProperty("test_interval"));
        st.add("test_iter", solver.getProperty("test_iter"));
        st.add("snapshot_prefix", solver.getProperty("snapshot_prefix"));

        st.add("train_data_itr", getIteratorName("TRAIN"));
        st.add("test_data_itr", getIteratorName("TEST"));

        String context = solver.getProperty("solver_mode", "cpu").toLowerCase();
        context = String.format("mx.%s()", context);
        st.add("ctx", context);

        st.add("loss", loss);

        st.add("data_names", getDataNames());
        st.add("label_names", getLabelNames());

        st.add("init_params", generateInitializer());

        st.add("init_optimizer", generateOptimizer());
        st.add("gamma", solver.getProperty("gamma"));
        st.add("power", solver.getProperty("power"));
        st.add("lr_update", generateLRUpdate());

        return st.render();
    }

    private String generateParamInitializer() {
        return gh.getTemplate("param_initializer").render();
    }

    private String generateMetricsClasses() {
        ST st = gh.getTemplate("metrics_classes");

        String display = solver.getProperty("display");
        String average_loss = solver.getProperty("average_loss");

        if (display != null) {
            st.add("display", display);
        }

        if (average_loss != null) {
            st.add("average_loss", average_loss);
        }

        return st.render();
    }

    private String generateParamsLoader() {
        return gh.getTemplate("params_loader").render();
    }

    private String getLoss(MLModel model, StringBuilder out) {
        List<String> losses = new ArrayList<>();
        for (Layer layer : model.getLayerList()) {
            if (layer.getType().toLowerCase().endsWith("loss")) {
                losses.add(gh.getVarname(layer.getTop()));
            }
        }

        if (losses.size() == 1) {
            return losses.get(0);
        } else if (losses.size() > 1) {
            String loss_var = "combined_loss";
            ST st = gh.getTemplate("group");
            st.add("var", loss_var);
            st.add("symbols", losses);
            out.append(st.render());
            return loss_var;
        } else {
            System.err.println("No loss found");
            return "unknown_loss";
        }
    }

    private String generateLRUpdate() {
        String code;
        String lrPolicy = solver.getProperty("lr_policy", "fixed").toLowerCase();
        ST st;
        switch (lrPolicy) {
            case "fixed":
                // lr stays fixed. No update needed
                code = "";
                break;
            case "multistep":
                st = gh.getTemplate("lrpolicy_multistep");
                st.add("steps", solver.getProperties("stepvalue"));
                code = st.render();
                break;
            case "step":
            case "exp":
            case "inv":
            case "poly":
            case "sigmoid":
                st = gh.getTemplate("lrpolicy_" + lrPolicy);
                code = st.render();
                break;
            default:
                String message = "Unknown lr_policy: " + lrPolicy;
                System.err.println(message);
                code = "# " + message + System.lineSeparator();
                break;
        }
        return Utils.indent(code, 2, true, 4);
    }

    private String generateValidationMetrics(MLModel mlModel) {
        return new AccuracyMetricsGenerator().generate(mlModel);
    }

    private String generateOptimizer() {
        Optimizer optimizer = new Optimizer(solver);
        return optimizer.generateInitCode();
    }

    private String generateInitializer() {
        ST st = gh.getTemplate("init_params");
        st.add("params_file", paramsFilePath);
        return st.render();
    }

    private String generateImports() {
        return gh.getTemplate("imports").render();
    }

    private StringBuilder generateIterators() {
        StringBuilder code = new StringBuilder();

        for (Layer layer : mlModel.getDataLayers()) {
            String iterator = generateIterator(layer);
            code.append(iterator);
        }

        return code;
    }

    private String getIteratorName(String phase) {
        for (Layer layer : mlModel.getDataLayers()) {
            String layerPhase = layer.getAttr("include.phase", phase);
            if (phase.equalsIgnoreCase(layerPhase)) {
                return layerPhase.toLowerCase() + "_" + layer.getName() + "_" + "itr";
            }
        }
        return null;
    }

    private List<String> getDataNames() {
        return getDataNames(0);
    }

    private List<String> getLabelNames() {
        return getDataNames(1);
    }

    private List<String> getDataNames(int topIndex) {
        List<String> dataList = new ArrayList<String>();
        for (Layer layer : mlModel.getDataLayers()) {
            if (layer.getAttr("include.phase").equalsIgnoreCase("train")) {
                String dataName = layer.getTops().get(topIndex);
                if (dataName != null) {
                    dataList.add(String.format("'%s'", dataName));
                }
            }
        }
        return dataList;
    }

    private StringBuilder generateInputVars() {
        StringBuilder code = new StringBuilder();

        Set<String> tops = new HashSet<String>();

        for (Layer layer : mlModel.getDataLayers())
            for (String top : layer.getTops())
                tops.add(top);

        for (String top : tops)
            code.append(gh.generateVar(gh.getVarname(top), top, null, null, null, null));

        code.append(System.lineSeparator());
        return code;
    }

    private String generateIterator(Layer layer) {
        String iteratorName = layer.getAttr("include.phase");
        iteratorName = iteratorName.toLowerCase();
        iteratorName = iteratorName + "_" + layer.getName() + "_" + "itr";

        ST st = stGroup.getInstanceOf("iterator");

        String prototxt = layer.getPrototxt();
        prototxt = prototxt.replace("\r", "");
        prototxt = prototxt.replace("\n", " \\\n");
        prototxt = "'" + prototxt + "'";
        prototxt = Utils.indent(prototxt, 1, true, 4);

        st.add("iter_name", iteratorName);
        st.add("prototxt", prototxt);

        String dataName = "???";
        if (layer.getTops().size() >= 1) {
            dataName = layer.getTops().get(0);
        } else {
            System.err.println(String.format("Data layer %s doesn't have data", layer.getName()));
        }
        st.add("data_name", dataName);

        String labelName = "???";
        if (layer.getTops().size() >= 1) {
            labelName = layer.getTops().get(1);
        } else {
            System.err.println(String.format("Data layer %s doesn't have label", layer.getName()));
        }
        st.add("label_name", labelName);

        if (layer.hasAttr("data_param.num_examples")) {
            st.add("num_examples", layer.getAttr("data_param.num_examples"));
        }

        return st.render();
    }

}
