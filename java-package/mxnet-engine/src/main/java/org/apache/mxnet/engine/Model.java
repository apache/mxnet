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

package org.apache.mxnet.engine;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.exception.MalformedModelException;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.nn.SymbolBlock;
import org.apache.mxnet.repository.Item;
import org.apache.mxnet.repository.Repository;
import org.apache.mxnet.translate.NoOpTranslator;
import org.apache.mxnet.translate.Translator;
import org.apache.mxnet.util.PairList;
import org.apache.mxnet.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A model is a collection of artifacts that is created by the training process.
 *
 * <p>Model contains methods to load and process a model object. In addition, it provides MXNet
 * Specific functionality, such as getSymbol to obtain the Symbolic graph and getParameters to
 * obtain the parameter NDArrays
 */
public class Model extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(Model.class);
    private static final int MODEL_VERSION = 1;

    protected Path modelDir;
    protected SymbolBlock symbolBlock;
    protected String modelName;
    protected DataType dataType;
    protected PairList<String, Shape> inputData;
    protected Map<String, Object> artifacts = new ConcurrentHashMap<>();
    protected Map<String, String> properties = new ConcurrentHashMap<>();

    Model(String name, Device device) {
        this(BaseMxResource.getSystemMxResource(), name, device);
    }

    private Model(MxResource parent, String name, Device device) {
        super(parent);
        setDevice(Device.defaultIfNull(device));
        setDataType(DataType.FLOAT32);
        setModelName(name);
    }

    /**
     * Create a default {@link Predictor} instance, with {@link NoOpTranslator} as default
     * translator , and do not copy parameters to parameter store
     *
     * @return {@link Predictor}
     */
    public Predictor<NDList, NDList> newPredictor() {
        Translator<NDList, NDList> noOpTranslator = new NoOpTranslator();
        return newPredictor(noOpTranslator, false);
    }

    /**
     * Create a default {@link Predictor} instance, with {@link NoOpTranslator} as default
     * translator
     *
     * @param copy whether to copy the parameters to the parameter store
     * @return {@link Predictor}
     */
    public Predictor<NDList, NDList> newPredictor(boolean copy) {
        Translator<NDList, NDList> noOpTranslator = new NoOpTranslator();
        return newPredictor(noOpTranslator, copy);
    }

    /**
     * Create {@link Predictor} instance, with specific {@link Translator} and {@code copy}
     *
     * @param translator {@link Translator} used to convert inputs and outputs into {@link NDList}
     *     to get inferred
     * @param copy whether to copy the parameters to the parameter store
     * @return {@link Predictor}
     */
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, boolean copy) {
        return new Predictor<>(this, translator, copy);
    }

    /**
     * Create and initialize a MxModel from the model directory
     *
     * @param modelPath {@code Path} model directory
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(Path modelPath) throws IOException {
        return loadModel("model", modelPath);
    }

    /**
     * Create and initialize a MxModel from repository Item
     *
     * @param modelItem {@link Item} model directory
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(Item modelItem) throws IOException {
        Model model = createModel(modelItem);
        model.initial();
        return model;
    }

    /**
     * Create and initialize a MxModel with a model name from the model directory
     *
     * @param modelName {@link String} model name
     * @param modelPath {@link Path} model directory
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(String modelName, Path modelPath) throws IOException {
        Model model = createModel(modelName, modelPath);
        model.initial();
        return model;
    }

    /**
     * Create a MxModel with specific model name and model directory. By default, the {@link Model}
     * instance is managed by the top level {@link BaseMxResource}
     *
     * @param modelName {@String} model name
     * @param modelDir {@Path} local model path
     * @throws IOException when IO operation fails in loading a resource
     */
    static Model createModel(String modelName, Path modelDir) {
        Model model = new Model(modelName, Device.defaultIfNull());
        model.setModelDir(modelDir);
        return model;
    }

    /**
     * Create a sample MxModel Download or find the local path for the sample model
     *
     * @param item {@@Item} sample model to be created
     * @throws IOException when IO operation fails in loading a resource
     */
    static Model createModel(Item item) throws IOException {
        Path modelDir = Repository.initRepository(item);
        return createModel(item.getName(), modelDir);
    }

    /**
     * Initialize the model object Download or find the path for target model Load parameters and
     * symbol from the path.
     *
     * @throws IOException when IO operation fails in loading a resource
     * @throws FileNotFoundException if Model Directory is not assigned
     */
    public void initial() throws IOException {
        if (getModelDir() == null) {
            throw new FileNotFoundException("Model path is not defined!");
        }
        load(getModelDir());
    }

    /**
     * Loads the model from the {@code modelPath}.
     *
     * @param modelPath the directory or file path of the model location
     * @throws IOException when IO operation fails in loading a resource
     * @throws MalformedModelException if model file is corrupted
     */
    public void load(Path modelPath) throws IOException, MalformedModelException {
        load(modelPath, null, null);
    }

    /**
     * Loads the MXNet model from a specified location.
     *
     * <p>MXNet Model looks for {MODEL_NAME}-symbol.json and {MODEL_NAME}-{EPOCH}.params files in
     * the specified directory. By default, It will pick up the latest epoch of the parameter file.
     * However, users can explicitly specify an epoch to be loaded:
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath the directory of the model
     * @param prefix the model file name or path prefix
     * @param options load model options, see documentation for the specific engine
     * @throws IOException Exception for file loading
     */
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        Path paramFile = paramPathResolver(prefix, options);
        if (paramFile == null) {
            prefix = modelDir.toFile().getName();
            paramFile = paramPathResolver(prefix, options);
            if (paramFile == null) {
                throw new FileNotFoundException(
                        "Parameter file with prefix: " + prefix + " not found in: " + modelDir);
            }
        }

        if (getMxSymbolBlock() == null) {
            // load MxSymbolBlock
            Path symbolFile = modelDir.resolve(prefix + "-symbol.json");
            if (Files.notExists(symbolFile)) {
                throw new FileNotFoundException(
                        "Symbol file not found: "
                                + symbolFile
                                + ", please set block manually for imperative model.");
            }

            // TODO: change default name "data" to model-specific one
            setMxSymbolBlock(SymbolBlock.createMxSymbolBlock(this, symbolFile));
        }
        loadParameters(paramFile, options);
        // TODO: Check if Symbol has all names that params file have
        if (options != null && options.containsKey("MxOptimizeFor")) {
            String optimization = (String) options.get("MxOptimizeFor");
            getMxSymbolBlock().optimizeFor(optimization, getDevice());
        }
    }

    protected Path paramPathResolver(String prefix, Map<String, ?> options) throws IOException {
        try {
            int epoch = getEpoch(prefix, options);
            return getModelDir()
                    .resolve(String.format(Locale.ROOT, "%s-%04d.params", prefix, epoch));
        } catch (FileNotFoundException e) {
            return null;
        }
    }

    private int getEpoch(String prefix, Map<String, ?> options) throws IOException {
        if (options != null) {
            Object epochOption = options.getOrDefault("epoch", null);
            if (epochOption != null) {
                return Integer.parseInt(epochOption.toString());
            }
        }
        return Utils.getCurrentEpoch(getModelDir(), prefix);
    }

    private void loadParameters(Path paramFile, Map<String, ?> options)
            throws IOException, MalformedModelException {

        NDList paramNDlist = JnaUtils.loadNdArray(this, paramFile, getDevice());

        List<Parameter> parameters = getMxSymbolBlock().getAllParameters();
        Map<String, Parameter> map = new LinkedHashMap<>();
        parameters.forEach(p -> map.put(p.getName(), p));

        for (NDArray nd : paramNDlist) {
            String key = nd.getName();
            if (key == null) {
                throw new IllegalArgumentException("Array names must be present in parameter file");
            }

            String paramName = key.split(":", 2)[1];
            Parameter parameter = map.remove(paramName);
            parameter.setArray(nd);
        }
        getMxSymbolBlock().setInputNames(new ArrayList<>(map.keySet()));

        // TODO: Find a better to infer model DataType from SymbolBlock.
        dataType = paramNDlist.head().getDataType();
        logger.debug("MXNet Model {} ({}) loaded successfully.", paramFile, dataType);
    }

    public Path getModelDir() {
        return modelDir;
    }

    public void setModelDir(Path modelDir) {
        this.modelDir = modelDir;
    }

    public SymbolBlock getMxSymbolBlock() {
        return symbolBlock;
    }

    public void setMxSymbolBlock(SymbolBlock symbolBlock) {
        this.symbolBlock = symbolBlock;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public DataType getDataType() {
        return dataType;
    }

    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    public PairList<String, Shape> getInputData() {
        return inputData;
    }

    public void setInputData(PairList<String, Shape> inputData) {
        this.inputData = inputData;
    }

    public Object getArtifact(String key) {
        return artifacts.get(key);
    }

    public void setArtifact(String key, Object artifact) {
        artifacts.put(key, artifact);
    }

    public String getProperty(String key) {
        return properties.get(key);
    }

    public void setProperties(String key, String property) {
        this.properties.put(key, property);
    }

    public Device getDevice() {
        if (device == null) {
            return super.getDevice();
        }
        return device;
    }

    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free Model instance: %S", this.getModelName()));
            // release sub resources
            super.freeSubResources();
            // release itself
            this.symbolBlock = null;
            this.artifacts = null;
            this.properties = null;
            setClosed();
            logger.debug(String.format("Finish to free Model instance: %S", this.getModelName()));
        }
    }
}
