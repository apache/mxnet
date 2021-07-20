package org.apache.mxnet.engine;

import org.apache.mxnet.exception.MalformedModelException;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.apache.mxnet.util.PairList;
import org.apache.mxnet.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MxModel extends MxResource{

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);
    private static final int MODEL_VERSION = 1;

    protected Path modelDir;
    protected MxSymbolBlock mxSymbolBlock;
    protected String modelName;
    protected DataType dataType;
    protected PairList<String, Shape> inputData;
    protected Map<String, Object> artifacts = new ConcurrentHashMap<>();
    protected Map<String, String> properties = new ConcurrentHashMap<>();

    MxModel(String name, Device device) {
        super(BaseMxResource.getSystemMxResource().newSubMxResource());
        setDevice(Device.defaultIfNull(device));
        setDataType(DataType.FLOAT32);
        setModelName(name);
    }

    /**
     * Loads the MXNet model from a specified location.
     *
     * <p>MXNet Model looks for {MODEL_NAME}-symbol.json and {MODEL_NAME}-{EPOCH}.params files in
     * the specified directory. By default, It will pick up the latest epoch of the
     * parameter file. However, users can explicitly specify an epoch to be loaded:
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
//    public void load(Path modelPath, String prefix, Map<String, ?> options)
//            throws IOException, MalformedModelException {
//        modelDir = modelPath.toAbsolutePath();
//        if (prefix == null) {
//            prefix = modelName;
//        }
//        Path paramFile = paramPathResolver(prefix, options);
//        if (paramFile == null) {
//            prefix = modelDir.toFile().getName();
//            paramFile = paramPathResolver(prefix, options);
//            if (paramFile == null) {
//                throw new FileNotFoundException(
//                        "Parameter file with prefix: " + prefix + " not found in: " + modelDir);
//            }
//        }
//
//        if (getMxSymbolBlock() == null) {
//            block = loadFromBlockFactory();
//        }
//
//        if (block == null) {
//            // load MxSymbolBlock
//            Path symbolFile = modelDir.resolve(prefix + "-symbol.json");
//            if (Files.notExists(symbolFile)) {
//                throw new FileNotFoundException(
//                        "Symbol file not found: "
//                                + symbolFile
//                                + ", please set block manually for imperative model.");
//            }
//            Symbol symbol =
//                    Symbol.load((MxNDManager) manager, symbolFile.toAbsolutePath().toString());
//            // TODO: change default name "data" to model-specific one
//            block = new MxSymbolBlock(manager, symbol);
//        }
//        loadParameters(paramFile, options);
//        // TODO: Check if Symbol has all names that params file have
//        if (options != null && options.containsKey("MxOptimizeFor")) {
//            String optimization = (String) options.get("MxOptimizeFor");
//            ((MxSymbolBlock) block).optimizeFor(optimization);
//        }
//    }

    protected Path paramPathResolver(String prefix, Map<String, ?> options) throws IOException {
        Object epochOption = null;
        if (options != null) {
            epochOption = options.get("epoch");
        }
        int epoch;
        if (epochOption == null) {
            epoch = Utils.getCurrentEpoch(modelDir, prefix);
            if (epoch == -1) {
                return null;
            }
        } else {
            epoch = Integer.parseInt(epochOption.toString());
        }

        return modelDir.resolve(String.format(Locale.ROOT, "%s-%04d.params", prefix, epoch));
    }

    public Path getModelDir() {
        return modelDir;
    }

    public void setModelDir(Path modelDir) {
        this.modelDir = modelDir;
    }

    public MxSymbolBlock getMxSymbolBlock() {
        return mxSymbolBlock;
    }

    public void setMxSymbolBlock(MxSymbolBlock mxSymbolBlock) {
        this.mxSymbolBlock = mxSymbolBlock;
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

}
