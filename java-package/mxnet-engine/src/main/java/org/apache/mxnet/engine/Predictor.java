package org.apache.mxnet.engine;

import org.apache.mxnet.exception.TranslateException;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Predictor<I, O> extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(Predictor.class);
    private Translator<I, O> translator;
    private long timestamp;
    private boolean prepared;
    private MxModel mxModel;
    protected ParameterStore parameterStore;

    /**
     * Creates a new instance of {@code Predictor} with the given {@link MxModel} and {@link
     * Translator}.
     *
     * @param mxModel the model on which the predictions are based
     * @param translator the translator to be used
     * @param copy whether to copy the parameters to the parameter store
     */
    public Predictor(MxModel mxModel, Translator<I, O> translator, boolean copy) {
        super(mxModel);
        this.mxModel = mxModel;
        this.translator = translator;
        this.parameterStore = new ParameterStore(getParent(), copy, mxModel.getDevice());
    }


    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public List<O> predict(List<I> input) {
        MxNDList[] ndLists = processInputs(input);
        for (int i = 0; i < ndLists.length; ++i) {
            ndLists[i] = forward(ndLists[i]);
        }
        return processOutPut(ndLists);
    }

    public O predict(I input) {
        return predict(Collections.singletonList(input)).get(0);
    }


    private MxNDList forward(MxNDList ndList) {
        logger.trace("Predictor input data: {}", ndList);
        return mxModel.getMxSymbolBlock().forward(parameterStore, ndList, false);
    }

    // TODO: add batch predict

    private MxNDList[] processInputs(List<I> inputs) throws TranslateException {
        int batchSize = inputs.size();
        MxNDList[] preprocessed = new MxNDList[batchSize];
        try {
            for (int i = 0; i < batchSize; ++i) {
                preprocessed[i] = translator.processInput(inputs.get(i));
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
        return preprocessed;
    }

    private List<O> processOutPut(MxNDList[] ndLists) throws TranslateException {
        List<O> outputs = new ArrayList<>();
        try {
            for (MxNDList mxNDList : ndLists) {
                outputs.add(translator.processOutput(mxNDList));
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
        return outputs;
    }
}
