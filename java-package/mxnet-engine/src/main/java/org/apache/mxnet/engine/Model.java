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
     * translator , and do not copy parameters to parameter store.
     *
     * @return {@link Predictor}
     */
    public Predictor<NDList, NDList> newPredictor() {
        Translator<NDList, NDList> noOpTranslator = new NoOpTranslator();
        return newPredictor(noOpTranslator);
    }

    /**
     * Create {@link Predictor} instance, with specific {@link Translator} and {@code copy}.
     *
     * @param translator {@link Translator} used to convert inputs and outputs into {@link NDList}
     *     to get inferred
     * @param <I> the input type
     * @param <O> the output type
     * @return {@link Predictor}
     */
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new Predictor<>(this, translator);
    }

    /**
     * Create and initialize a MxModel from the model directory.
     *
     * @param modelPath {@code Path} model directory
     * @return loaded {@code Model} instance
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(Path modelPath) throws IOException {
        return loadModel("model", modelPath);
    }

    /**
     * Create and initialize a MxModel from repository Item.
     *
     * @param modelItem {@link Item} model directory
     * @return {@link Model}
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(Item modelItem) throws IOException {
        Model model = createModel(modelItem);
        model.initial();
        return model;
    }

    /**
     * Create and initialize a MxModel with a model name from the model directory.
     *
     * @param modelName {@link String} model name
     * @param modelPath {@link Path} model directory
     * @return {@link Model}
     * @throws IOException when IO operation fails in loading a resource
     */
    public static Model loadModel(String modelName, Path modelPath) throws IOException {
        Model model = createModel(modelName, modelPath);
        model.initial();
        return model;
    }

    /**
     * Create a MxModel with specific model name and model directory. By default, the {@link Model}
     * instance is managed by the top level {@link BaseMxResource}.
     *
     * @param modelName {@String} model name
     * @param modelDir {@Path} local model path
     * @return {@link Model}
     * @throws IOException when IO operation fails in loading a resource
     */
    static Model createModel(String modelName, Path modelDir) {
        Model model = new Model(modelName, Device.defaultIfNull());
        model.setModelDir(modelDir);
        return model;
    }

    /**
     * Create a sample MxModel Download or find the local path for the sample model.
     *
     * @param item {@link Item} sample model to be created
     * @return created {@link Model} instance
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
     */
    public void load(Path modelPath) throws IOException {
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
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
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

        if (getSymbolBlock() == null) {
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
        loadParameters(paramFile);
        // TODO: Check if Symbol has all names that params file have
        if (options != null && options.containsKey("MxOptimizeFor")) {
            String optimization = (String) options.get("MxOptimizeFor");
            getSymbolBlock().optimizeFor(optimization, getDevice());
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

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    private void loadParameters(Path paramFile) {

        NDList paramNDlist = JnaUtils.loadNdArray(this, paramFile, getDevice());

        List<Parameter> parameters = getSymbolBlock().getAllParameters();
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
        getSymbolBlock().setInputNames(new ArrayList<>(map.keySet()));

        // TODO: Find a better to infer model DataType from SymbolBlock.
        dataType = paramNDlist.head().getDataType();
        logger.debug("MXNet Model {} ({}) loaded successfully.", paramFile, dataType);
    }

    /**
     * Get the modelDir from the Model.
     *
     * @return {@link Path} modelDir for the Model
     */
    public Path getModelDir() {
        return modelDir;
    }

    /**
     * Set the modelDir for the Model.
     *
     * @param modelDir {@link Path}
     */
    public void setModelDir(Path modelDir) {
        this.modelDir = modelDir;
    }

    /**
     * Get the symbolBlock of the Model.
     *
     * @return {@link SymbolBlock}
     */
    public SymbolBlock getSymbolBlock() {
        return symbolBlock;
    }

    /**
     * Set the symbolBlock for the Model.
     *
     * @param symbolBlock {@link SymbolBlock}
     */
    public void setMxSymbolBlock(SymbolBlock symbolBlock) {
        this.symbolBlock = symbolBlock;
    }

    /**
     * Get the name of the Model.
     *
     * @return modelName
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Set the model name for the Model.
     *
     * @param modelName for the Model
     */
    public final void setModelName(String modelName) {
        this.modelName = modelName;
    }

    /**
     * Get data type for the Model.
     *
     * @return {@link DataType}
     */
    public DataType getDataType() {
        return dataType;
    }

    /**
     * Set data type for the Model.
     *
     * @param dataType {@link DataType}
     */
    public final void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    /**
     * Get input data of the Model.
     *
     * @return {@link PairList} inputData
     */
    public PairList<String, Shape> getInputData() {
        return inputData;
    }

    /**
     * Set input data for the Model.
     *
     * @param inputData {@link PairList}
     */
    public void setInputData(PairList<String, Shape> inputData) {
        this.inputData = inputData;
    }

    /**
     * Get the Artifact Object from artifacts by key.
     *
     * @param key for the Artifact Object
     * @return Artifact {@link Object} instance
     */
    public Object getArtifact(String key) {
        return artifacts.get(key);
    }

    /**
     * Set the Artifact Object for artifacts.
     *
     * @param key for the Artifact
     * @param artifact {@link Object}
     */
    public void setArtifact(String key, Object artifact) {
        artifacts.put(key, artifact);
    }

    /**
     * Get the property from properties by key.
     *
     * @param key {@link String}
     * @return {@link String} property
     */
    public String getProperty(String key) {
        return properties.get(key);
    }

    /**
     * Set the property for the Model.
     *
     * @param key for the property
     * @param property value of the property
     */
    public void setProperties(String key, String property) {
        this.properties.put(key, property);
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (device == null) {
            return super.getDevice();
        }
        return device;
    }

    /** {@inheritDoc} */
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
            setClosed(true);
            logger.debug(String.format("Finish to free Model instance: %S", this.getModelName()));
        }
    }
}
