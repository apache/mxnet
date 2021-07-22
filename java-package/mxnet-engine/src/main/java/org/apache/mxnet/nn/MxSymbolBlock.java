package org.apache.mxnet.nn;

import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.MxResourceList;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.exception.MalformedModelException;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MxSymbolBlock extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(MxSymbolBlock.class);

    /** The shape of the input for this block, set by the initialization process. */
    protected Shape[] inputShapes;

    /** List of names for the input, named inputs should be manually set in sub class. */
    protected List<String> inputNames = Collections.emptyList();

    /**
     * The model version of this block, used for checking if parameters are still valid during
     * parameter loading.
     */
    protected byte version;

    /**
     * All direct parameters of this Block. Keys are name of the parameters.
     *
     * <p>Use the {@link MxSymbolBlock#addParameter(Parameter)} method to add children. All
     * parameters in this map are automatically loaded / saved.
     */
    protected LinkedHashMap<String, Parameter> parameters = new LinkedHashMap<>();

    private static final byte VERSION = 3;

    private CachedOp op;
    private Symbol symbol;
    private List<Parameter> mxNetParams; // includes input data
    private Map<String, Shape> paramShapes;
    private Shape[] outputShapes;
    private PairList<String, Shape> inputDescriptions;
    private PairList<String, Shape> outputDescriptions;
    private boolean first;


    /**
     * Constructs a {@code MxSymbolBlock} for a {@link Symbol}.
     *
     * @param parent the parent MxResource to use for the block
     * @param symbol the symbol containing the block's symbolic graph
     */
    public MxSymbolBlock(MxResource parent, Symbol symbol) {
        super();
        setParent(parent);
        this.symbol = symbol;
        initBlock();
    }

    /**
     * Constructs an empty {@code MxSymbolBlock}.
     *
     * @param parent the parent {@code MxSymbolBlock} instance to manage this MxSymbolBlock
     */
    private MxSymbolBlock(MxResource parent) {
        super();
        setParent(parent);
    }

    /**
     * Constructs an {@code MxSymbolBlock} and load the symbol according to {@code Path}
     * The life circle of the {@code Symbol} instance is managed by parent {@codd MxResource}.
     * @param parent the parent MxResource Object to manage this MxSymbolBlock
     * @param symbolPath the Path to load symbol
     */
    public static MxSymbolBlock createMxSymbolBlock(MxResource parent, Path symbolPath) {
        MxSymbolBlock mxSymbolBlock = new MxSymbolBlock(parent);
        mxSymbolBlock.loadSymbol(symbolPath);
        mxSymbolBlock.initBlock();
        return mxSymbolBlock;
    }

    private void loadSymbol(Path symbolPath) {
        Symbol symbol = Symbol.loadSymbol(this, symbolPath);
        this.symbol = symbol;
    }

    /**
     * Sets the names of the input data.
     *
     * @param inputNames the names of the input data
     */
    public void setInputNames(List<String> inputNames) {
        this.inputNames = inputNames;
        // now that we know which of the parameters are just input placeholders and which
        // are trainable, add them properly so they are correctly handled
        Set<String> nameLookup = new HashSet<>(inputNames);
        for (Parameter mxNetParameter : mxNetParams) {
            if (!nameLookup.contains(mxNetParameter.getName())) {
                addParameter(mxNetParameter);
            }
        }
    }

    protected final Parameter addParameter(Parameter parameter) {
        parameters.put(parameter.getName(), parameter);
        return parameter;
    }

    /**
     * Returns the list of inputs and parameter NDArrays.
     *
     * @return the list of inputs and parameter NDArrays
     */
    public List<Parameter> getAllParameters() {
        return mxNetParams;
    }

    /**
     * Returns the layers' name.
     *
     * @return a List of String containing the layers' name
     */
    public List<String> getLayerNames() {
        return symbol.getLayerNames();
    }

    /**
     * Returns the Symbolic graph from the model.
     *
     * @return a {@link Symbol} object
     */
    public Symbol getSymbol() {
        return symbol;
    }

    /**
     * Applies Optimization algorithm for the model.
     *
     * @param optimization the name of the optimization
     */
    public void optimizeFor(String optimization, Device device) {
        Symbol newSymbol = symbol.optimizeFor(optimization, device);
        symbol.close();
        symbol = newSymbol;
    }

    public PairList<String, Shape> describeInput() {
        if (inputDescriptions == null) {
            inputDescriptions = new PairList<>();
            for (String name : inputNames) {
                // Add empty shapes as input shapes are not saved
                // in MXNet models
                logger.warn(
                        "Input shapes are unknown, please run predict or forward once"
                                + "and call describeInput again.");
                inputDescriptions.add(name, new Shape());
            }
        }
        return inputDescriptions;
    }

    public PairList<String, Shape> describeOutput() {
        if (outputDescriptions == null) {
            logger.warn(
                    "Output shapes are unknown, please run predict or forward once"
                            + "and call describeOutput again.");
        }
        return outputDescriptions;
    }

    /**
     * Applies the operating function of the mxSymbolBlock once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass
     * @param params optional parameters
     * @param device device to use
     * @return the output of the forward pass
     */
    public final MxNDList forward(
            ParameterStore parameterStore,
            MxNDList inputs,
            boolean training,
            PairList<String, Object> params,
            Device device) {

        if (!isInitialized()) {
            initialize(getParent(), DataType.FLOAT32, device, inputs.getShapes());
        }
        return forwardInternal(parameterStore, inputs, training, params);
    }

    /**
     * Applies the operating function of the block once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass
     * @return the output of the forward pass
     */
    public MxNDList forward(ParameterStore parameterStore, MxNDList inputs, boolean training) {
        return forward(parameterStore, inputs, training, null, getDevice());
    }

    /**
     * A forward call using both training data and labels.
     *
     * <p>Within this forward call, it can be assumed that training is true.
     *
     * @param parameterStore the parameter store
     * @param data the input data NDList
     * @param labels the input labels NDList
     * @param params optional parameters
     * @return the output of the forward pass
     * @see #forward(ParameterStore, MxNDList, boolean, PairList, Device)
     */
    public MxNDList forward(
            ParameterStore parameterStore,
            MxNDList data,
            MxNDList labels,
            PairList<String, Object> params,
            Device device) {
        if (!isInitialized()) {
            initialize(getParent(), DataType.FLOAT32, device, data.getShapes());
        }
        return forwardInternal(parameterStore, data, labels, params);
    }

    /**
     * A helper for {@link MxSymbolBlock#forward(ParameterStore, MxNDList, MxNDList, PairList, Device)} after
     * initialization.
     *
     * @param parameterStore the parameter store
     * @param data the input data NDList
     * @param labels the input labels NDList
     * @param params optional parameters
     * @return the output of the forward pass
     * @see #forward(ParameterStore, MxNDList, boolean, PairList, Device)
     */
    protected MxNDList forwardInternal(
            ParameterStore parameterStore,
            MxNDList data,
            MxNDList labels,
            PairList<String, Object> params) {
        return forwardInternal(parameterStore, data, true, params);
    }

    public boolean isInitialized() {
        for (Parameter param : getParameters().values()) {
            if (!param.isInitialized()) {
                return false;
            }
        }
        return true;
    }

    public void initialize(MxResource parent, DataType dataType, Device device, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        // if parameters are initialized, skip it
        if (!isInitialized()) {
            // setShape for all params
//            prepare(inputShapes);
            // do nothing
        }
        for (Parameter parameter : parameters.values()) {
            parameter.initialize(parent, dataType, device);
        }
        initializeChildBlocks();
    }

    /**
     * Initializes the Child blocks of this block. You need to override this method if your subclass
     * has child blocks. Used to determine the correct input shapes for child blocks based on the
     * requested input shape for this block.
     */
    protected void initializeChildBlocks() {
        if (!getSubResource().isEmpty()) {
            throw new IllegalStateException(
                    getClass().getSimpleName()
                            + " has child blocks but initializeChildBlocks is not overwritten.");
        }
    }

    protected void beforeInitialize(Shape... inputShapes) {
        if (inputNames.isEmpty()) {
            // automatically assign input names
            inputNames = new ArrayList<>();
            for (int i = 0; i < inputShapes.length; ++i) {
                inputNames.add("data" + i);
            }
        }
        this.inputShapes = inputShapes;
    }

    public ParameterList getParameters() {
        // we accumulate a list of all parameters by starting with a list of the direct parameters
        ParameterList allParams = getDirectParameters();
        // then we add the parameters of child blocks
        for (Pair<String, MxResource> childPair : getChildren()) {
            if (MxSymbolBlock.class.equals(childPair.getValue().getClass())) {
                MxSymbolBlock mxSymbolBlock = (MxSymbolBlock) childPair.getValue();
                for (Pair<String, Parameter> paramPair : mxSymbolBlock.getParameters()) {
                    // we prepend the name of the child block to the parameter name
                    allParams.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
                }
            }
        }
        return allParams;
    }

    public MxResourceList getChildren() {
        MxResourceList defensiveCopy = new MxResourceList(getSubResource().size());
        for (Map.Entry<String, MxResource> entry : getSubResource().entrySet()) {
            defensiveCopy.add(entry.getKey(), entry.getValue());
        }
        return defensiveCopy;
    }

    public ParameterList getDirectParameters() {
        return new ParameterList(parameters);
    }

    protected MxNDList forwardInternal(
            ParameterStore parameterStore,
            MxNDList inputs,
            boolean training,
            PairList<String, Object> params) {
        if (first) {
            synchronized (MxSymbolBlock.class) {
                if (first) {
                    // create CachedOp is not thread-safe
                    // add synchronized block to avoid creating multiple CachedOps
                    op = JnaUtils.createCachedOp(this, getParent(), training);
                    inputDescriptions = new PairList<>();
                    outputDescriptions = new PairList<>();
                    for (MxNDArray array : inputs) {
                        inputDescriptions.add(array.getName(), array.getShape());
                    }
                    MxNDList outputs = op.forward(parameterStore, inputs, training);
                    for (MxNDArray array : outputs) {
                        outputDescriptions.add(array.getName(), array.getShape());
                    }
                    first = false;
                    return outputs;
                }
            }
        }
        return op.forward(parameterStore, inputs, training);
    }

    public Shape[] getOutputShapes(Shape[] inputShapes) {
        if (outputShapes == null) {
            String[] outputNames = symbol.getOutputNames();
            outputShapes = new Shape[outputNames.length];
            for (int i = 0; i < outputShapes.length; ++i) {
                outputShapes[i] = getParameterShape(outputNames[i], inputShapes);
            }
        }
        return outputShapes;
    }

    public void removeLastBlock() {
        List<String> layerNames = getLayerNames();
        String layerName = layerNames.get(layerNames.size() - 2);

        Symbol sliced = symbol.get(layerName);
        symbol.close();
        symbol = sliced;

        HashSet<String> set = new HashSet<>(Arrays.asList(symbol.getAllNames()));
        for (int i = mxNetParams.size() - 1; i >= 0; --i) {
            Parameter parameter = mxNetParams.get(i);
            if (!set.contains(parameter.getName())) {
                mxNetParams.remove(i).close();
                parameters.remove(parameter.getName(), parameter);
            }
        }
    }

    private Shape getParameterShape(String name, Shape[] inputShapes) {
        if (paramShapes == null) {
            PairList<String, Shape> pairs = new PairList<>();
            for (int i = 0; i < inputNames.size(); i++) {
                pairs.add(inputNames.get(i), inputShapes[i]);
            }
            paramShapes = symbol.inferShape(pairs);
        }
        if (paramShapes.containsKey(name)) {
            return paramShapes.get(name);
        } else {
            throw new IllegalArgumentException("Name " + name + " not found");
        }
    }

    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        String json = symbol.toJsonString();
        // symbol size may go beyond os.writeUTF() size (65535)
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        os.writeInt(bytes.length);
        os.write(bytes);
        int size = inputNames.size();
        os.writeInt(size);
        for (String name : inputNames) {
            os.writeUTF(name);
        }
        for (Parameter parameter : mxNetParams) {
            parameter.save(os);
        }
    }

    public void loadParameters(MxResource parent, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version > VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        if (version < VERSION && symbol == null) {
            throw new IllegalStateException(
                    "Symbol is required for version 2, please use Model to load");
        }
        if (version == VERSION) {
            int len = is.readInt();
            byte[] bytes = new byte[len];
            if (is.read(bytes) == -1) {
                throw new MalformedModelException("InputStream ends at symbol loading!");
            }
            // init block only if it is not set
            symbol =
                    Symbol.loadJson(this, new String(bytes, StandardCharsets.UTF_8));
            initBlock();
        }
        int size = is.readInt();
        for (int i = 0; i < size; ++i) {
            inputNames.add(is.readUTF());
        }

        for (Parameter parameter : mxNetParams) {
            parameter.load(parent, is);
        }
        setInputNames(inputNames);
    }

    private void initBlock() {
        inputNames = new ArrayList<>();

        String[] allNames = symbol.getAllNames();
        mxNetParams = new ArrayList<>(allNames.length);

        Set<String> auxNameSet = new HashSet<>(Arrays.asList(symbol.getAuxNames()));
        for (String name : allNames) {
            Parameter.Type type = inferType(name);
            boolean requireGrad = !auxNameSet.contains(name);
            mxNetParams.add(
                    Parameter.builder()
                            .setName(name)
                            .setType(type)
                            .optRequiresGrad(requireGrad)
                            .build());
        }
        first = true;
    }

    private static Parameter.Type inferType(String name) {
        if (name.endsWith("bias")) {
            return Parameter.Type.BIAS;
        } else if (name.endsWith("gamma")) {
            return Parameter.Type.GAMMA;
        } else if (name.endsWith("beta")) {
            return Parameter.Type.BETA;
        } else if (name.endsWith("moving_mean") || name.endsWith("running_mean")) {
            return Parameter.Type.RUNNING_MEAN;
        } else if (name.endsWith("moving_var") || name.endsWith("running_var")) {
            return Parameter.Type.RUNNING_VAR;
        } else if (name.endsWith("weight")) {
            return Parameter.Type.WEIGHT;
        }
        return Parameter.Type.OTHER;
    }
}
