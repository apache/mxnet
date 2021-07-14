package org.apache.mxnet.engine;

import org.apache.mxnet.ndarray.types.Shape;

import java.util.Arrays;

public class MxNDArrayEx {
    
    private static final MxNDArrayIndexer INDEXER = new MxNDArrayIndexer();

    private MxNDArray array;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link MxNDArray}.
     *
     * @param parent the {@link MxNDArray} to extend
     */
    MxNDArrayEx(MxNDArray parent) {
        this.array = parent;
    }

    // TODO only used to calculate zero-dim numpy shape
    // remove it once MXNet have all the np op that we support
    private Shape deriveBroadcastedShape(Shape lhs, Shape rhs) {
        long[] result = new long[Math.max(lhs.dimension(), rhs.dimension())];
        long lDiff = result.length - lhs.dimension();
        long rDiff = result.length - rhs.dimension();
        for (int i = 0; i < result.length; i++) {
            long l = 1;
            long r = 1;
            if (i >= lDiff) {
                l = lhs.get(Math.toIntExact(i - lDiff));
            }
            if (i >= rDiff) {
                r = rhs.get(Math.toIntExact(i - rDiff));
            }
            if (l != r) {
                if (l != 1 && r != 1) {
                    throw new IllegalArgumentException(
                            "operands could not be broadcast together with shapes "
                                    + lhs
                                    + " "
                                    + rhs);
                }
                result[i] = (l == 1) ? r : l;
            } else {
                result[i] = l;
            }
        }
        return new Shape(result);
    }

    ////////////////////////////////////////
    // MxNDArrays
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public MxNDArray rdiv(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_rdiv_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rdiv(MxNDArray b) {
        return b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rdivi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_rdiv_scalar", new MxNDArray[] {array}, new MxNDArray[] {array}, params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rdivi(MxNDArray b) {
        getManager().invoke("elemwise_div", new MxNDArray[] {b, array}, new MxNDArray[] {array}, null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rsub(Number n) {
        return array.sub(n).neg();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rsub(MxNDArray b) {
        return array.sub(b).neg();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rsubi(Number n) {
        return array.subi(n).negi();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rsubi(MxNDArray b) {
        return array.subi(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rmod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_rmod_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rmod(MxNDArray b) {
        return b.mod(array);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rmodi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager()
                .invoke("_npi_rmod_scalar", new MxNDArray[] {array}, new MxNDArray[] {array}, params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rmodi(MxNDArray b) {
        getManager().invoke("_npi_mod", new MxNDArray[] {b, array}, new MxNDArray[] {array}, null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rpow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_rpower_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray rpowi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager()
                .invoke("_npi_rpower_scalar", new MxNDArray[] {array}, new MxNDArray[] {array}, params);
        return array;
    }

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public MxNDArray relu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "relu");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray sigmoid() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "sigmoid");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray tanh() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "tanh");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray softPlus() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softrelu");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray softSign() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "softsign");
        return getManager().invoke("_npx_activation", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray leakyRelu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "leaky");
        params.addParam("slope", alpha);
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray elu(float alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "elu");
        params.addParam("slope", alpha);
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray selu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "selu");
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray gelu() {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "gelu");
        return getManager().invoke("_npx_leaky_relu", array, params);
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public MxNDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "max");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray globalMaxPool() {
        MxOpParams params = new MxOpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "max");
        params.addParam("global_pool", true);
        try (MxNDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "avg");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        params.addParam("count_include_pad", countIncludePad);
        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray globalAvgPool() {
        MxOpParams params = new MxOpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "avg");
        params.addParam("global_pool", true);
        try (MxNDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer instead");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("p_value", (int) normType);
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "lp");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");

        return getManager().invoke("_npx_pooling", getArray(), params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray globalLpPool(float normType) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer instead");
        }
        MxOpParams params = new MxOpParams();
        params.add("pool_type", "lp");
        params.addParam("p_value", (int) normType);
        params.addParam("global_pool", true);
        try (MxNDArray temp = getManager().invoke("_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public void adadeltaUpdate(
            NDList inputs,
            NDList weights,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float epsilon) {
        MxNDArray weight = inputs.get(0);
        MxNDArray grad = inputs.get(1);
        MxNDArray s = inputs.get(2);
        MxNDArray delta = inputs.get(3);

        // create a baseManager to close all intermediate MxNDArrays
        try (NDManager subManager = NDManager.newBaseManager()) {
            subManager.tempAttachAll(inputs, weights);

            // Preprocess Gradient
            grad.muli(rescaleGrad);
            if (clipGrad > 0) {
                grad = grad.clip(-clipGrad, clipGrad);
            }
            grad.addi(weight.mul(weightDecay));

            // Update s, g, and delta
            s.muli(rho).addi(grad.square().mul(1 - rho));
            MxNDArray g = delta.add(epsilon).sqrt().div(s.add(epsilon).sqrt()).mul(grad);
            delta.muli(rho).addi(g.square().mul(1 - rho));

            // Update weight
            weight.subi(g);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("epsilon", epsilon);

        getManager().invoke("adagrad_update", inputs, weights, params);
    }

    /** {@inheritDoc} */
    @Override
    public void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("beta1", beta1);
        params.addParam("beta2", beta2);
        params.addParam("epsilon", epsilon);
        params.addParam("lazy_update", lazyUpdate);

        getManager().invoke("adam_update", inputs, weights, params);
    }

    /** {@inheritDoc} */
    @Override
    public void rmspropUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float gamma1,
            float gamma2,
            float epsilon,
            boolean centered) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("gamma1", gamma1);
        params.addParam("epsilon", epsilon);

        if (!centered) {
            getManager().invoke("rmsprop_update", inputs, weights, params);
        } else {
            params.addParam("gamma2", gamma2);

            getManager().invoke("rmspropalex_update", inputs, weights, params);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("momentum", momentum);
        getManager().invoke("nag_mom_update", inputs, weights, params);
    }

    /** {@inheritDoc} */
    @Override
    public void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("lazy_update", lazyUpdate);

        if (momentum != 0) {
            params.addParam("momentum", momentum);
            getManager().invoke("sgd_mom_update", inputs, weights, params);
        } else {
            getManager().invoke("sgd_update", inputs, weights, params);
        }
    }

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public NDList convolution(
            MxNDArray input,
            MxNDArray weight,
            MxNDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", weight.getShape().slice(2));
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.addParam("dilate", dilation);
        params.addParam("num_group", groups);
        params.addParam("num_filter", weight.getShape().get(0));

        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            params.add("no_bias", false);
            inputs.add(bias);
        } else {
            params.add("no_bias", true);
        }

        return getManager().invoke("_npx_convolution", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList deconvolution(
            MxNDArray input,
            MxNDArray weight,
            MxNDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        MxOpParams params = new MxOpParams();
        params.addParam("kernel", weight.getShape().slice(2));
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.addParam("adj", outPadding);
        params.addParam("dilate", dilation);
        params.addParam("num_group", groups);
        params.addParam("num_filter", weight.getShape().get(0));

        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            params.add("no_bias", false);
            inputs.add(bias);
        } else {
            params.add("no_bias", true);
        }

        return getManager().invoke("_npx_deconvolution", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList linear(MxNDArray input, MxNDArray weight, MxNDArray bias) {
        MxOpParams params = new MxOpParams();
        params.addParam("num_hidden", weight.size(0));
        params.addParam("flatten", false);
        params.addParam("no_bias", bias == null);
        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            inputs.add(bias);
        }

        return getManager().invoke("_npx_fully_connected", inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(MxNDArray input, MxNDArray weight, SparseFormat sparse) {
        if (!sparse.equals(SparseFormat.DENSE) && !sparse.equals(SparseFormat.ROW_SPARSE)) {
            throw new IllegalArgumentException("MXNet only supports row sparse");
        }
        MxOpParams params = new MxOpParams();
        long inputDim = weight.getShape().get(0);
        long outputDim = weight.getShape().get(1);
        params.addParam("input_dim", inputDim);
        params.addParam("output_dim", outputDim);
        params.addParam("sparse_grad", sparse.getValue());
        return getManager().invoke("_npx_embedding", new NDList(input, weight), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(MxNDArray input, MxNDArray alpha) {
        MxOpParams params = new MxOpParams();
        params.addParam("act_type", "prelu");
        return getManager().invoke("_npx_leaky_relu", new NDList(input, alpha), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(MxNDArray input, float rate, boolean training) {
        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of dropout in MXNet should align with the mode of GradientCollector");
        }

        MxOpParams params = new MxOpParams();
        params.addParam("p", rate);

        return getManager().invoke("_npx_dropout", new NDList(input), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchNorm(
            MxNDArray input,
            MxNDArray runningMean,
            MxNDArray runningVar,
            MxNDArray gamma,
            MxNDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("fix_gamma", gamma == null);
        params.addParam("eps", eps);
        params.addParam("momentum", momentum);

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of batchNorm in MXNet should align with the mode of GradientCollector");
        }

        return getManager()
                .invoke(
                        "_npx_batch_norm",
                        new NDList(input, gamma, beta, runningMean, runningVar),
                        params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList rnn(
            MxNDArray input,
            MxNDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of rnn in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", state.getShape().tail());
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("state_outputs", true);
        opParams.addParam("mode", activation == RNN.Activation.TANH ? "rnn_tanh" : "rnn_relu");

        NDList inputs = new NDList();
        inputs.add(input);

        try (NDList temp = new NDList()) {
            for (MxNDArray param : params) {
                temp.add(param.flatten());
            }
            MxNDArray tempParam = MxNDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }

        inputs.add(state);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (MxNDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList gru(
            MxNDArray input,
            MxNDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of gru in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", state.getShape().tail());
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("state_outputs", true);
        opParams.addParam("mode", "gru");

        NDList inputs = new NDList();
        inputs.add(input);

        try (NDList temp = new NDList()) {
            for (MxNDArray param : params) {
                temp.add(param.flatten());
            }
            MxNDArray tempParam = MxNDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }

        inputs.add(state);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (MxNDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList lstm(
            MxNDArray input,
            NDList states,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
        Preconditions.checkArgument(
                params.size() == numParams,
                "The size of Params is incorrect expect "
                        + numParams
                        + " parameters but got "
                        + params.size());

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of lstm in MXNet should align with the mode of GradientCollector");
        }

        if (batchFirst) {
            input = input.swapAxes(0, 1);
        }

        MxOpParams opParams = new MxOpParams();
        opParams.addParam("mode", "lstm");
        opParams.addParam("p", dropRate);
        opParams.addParam("state_size", states.head().getShape().tail());
        opParams.addParam("state_outputs", true);
        opParams.addParam("num_layers", numLayers);
        opParams.addParam("bidirectional", bidirectional);
        opParams.addParam("lstm_state_clip_nan", true);

        NDList inputs = new NDList();
        inputs.add(input);
        try (NDList temp = new NDList()) {
            for (MxNDArray param : params) {
                temp.add(param.flatten());
            }
            MxNDArray tempParam = MxNDArrays.concat(temp);
            tempParam.attach(input.getManager());
            inputs.add(tempParam);
        }
        inputs.addAll(states);

        if (!batchFirst) {
            return getManager().invoke("_npx_rnn", inputs, opParams);
        }

        NDList result = getManager().invoke("_npx_rnn", inputs, opParams);
        try (MxNDArray temp = result.head()) {
            return new NDList(temp.swapAxes(0, 1), result.get(1), result.get(2));
        }
    }

    ////////////////////////////////////////
    // Image and CV
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public MxNDArray normalize(float[] mean, float[] std) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("mean", mean);
        params.addTupleParam("std", std);
        return getManager().invoke("_npx__image_normalize", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray toTensor() {
        return getManager().invoke("_npx__image_to_tensor", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray resize(int width, int height, int interpolation) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty MxNDArray");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("size", width, height);
        params.addParam("interp", interpolation);
        return getManager().invoke("_npx__image_resize", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray crop(int x, int y, int width, int height) {
        MxOpParams params = new MxOpParams();
        params.add("x", x);
        params.add("y", y);
        params.add("width", width);
        params.add("height", height);
        return getManager().invoke("_npx__image_crop", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray randomFlipLeftRight() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipLeftRight is not supported on GPU");
        }
        return getManager().invoke("_npx__image_random_flip_left_right", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray randomFlipTopBottom() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipTopBottom is not supported on GPU");
        }
        return getManager().invoke("_npx__image_random_flip_top_bottom", array, null);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray randomBrightness(float brightness) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomBrightness is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        float min = Math.max(0, 1 - brightness);
        float max = 1 + brightness;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return getManager().invoke("_npx__image_random_brightness", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray randomHue(float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomHue is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        float min = Math.max(0, 1 - hue);
        float max = 1 + hue;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return getManager().invoke("_npx__image_random_hue", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray randomColorJitter(
            float brightness, float contrast, float saturation, float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomColorJitter is not supported on GPU");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("brightness", brightness);
        params.addParam("contrast", contrast);
        params.addParam("saturation", saturation);
        params.addParam("hue", hue);
        return getManager().invoke("_npx__image_random_color_jitter", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArrayIndexer getIndexer() {
        return INDEXER;
    }

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.UseTryWithResources")
    public MxNDArray where(MxNDArray condition, MxNDArray other) {
        MxNDArray array1;
        MxNDArray array2;
        condition =
                (condition.getDataType() == DataType.BOOLEAN)
                        ? condition.toType(DataType.INT32, false)
                        : condition;
        if (array.getDataType() != other.getDataType()) {
            throw new IllegalArgumentException(
                    "DataType mismatch, required "
                            + array.getDataType()
                            + " actual "
                            + other.getDataType());
        }
        if (!array.shapeEquals(other)) {
            Shape res = deriveBroadcastedShape(array.getShape(), other.getShape());
            array1 = (!res.equals(array.getShape())) ? array.broadcast(res) : array;
            array2 = (!res.equals(other.getShape())) ? other.broadcast(res) : other;
        } else {
            array1 = array;
            array2 = other;
        }
        try {
            return getManager().invoke("where", new MxNDArray[] {condition, array1, array2}, null);
        } finally {
            if (array1 != array) {
                array1.close();
            }
            if (array2 != other) {
                array2.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray stack(NDList arrays, int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        MxNDArray[] srcArray = new MxNDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new MxNDArray[0]), 0, srcArray, 1, arrays.size());
        return getManager().invoke("_npi_stack", srcArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        MxOpParams params = new MxOpParams();
        // MXNet backend use dim as argument name
        params.addParam("axis", axis);
        MxNDArray[] srcArray = new MxNDArray[list.size() + 1];
        srcArray[0] = array;
        System.arraycopy(list.toArray(new MxNDArray[0]), 0, srcArray, 1, list.size());
        return getManager().invoke("_npi_concatenate", srcArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("minimum_negative_samples", minNegativeSamples);
        parameters.add("overlap_threshold", iouThreshold);
        parameters.add("ignore_label", ignoreLabel);
        parameters.add("negative_mining_ratio", negativeMiningRatio);
        parameters.add("negative_mining_thresh", negativeMiningThreshold);
        return getManager().invoke("MultiBoxTarget", inputs, parameters);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("sizes", sizes);
        parameters.add("ratios", ratios);
        parameters.add("steps", steps);
        parameters.add("offsets", offsets);
        parameters.add("clip", clip);
        return getManager().invoke("MultiBoxPrior", new NDList(array), parameters);
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreashold,
            boolean forceSuppress,
            int nmsTopK) {
        MxOpParams parameters = new MxOpParams();
        parameters.add("clip", clip);
        parameters.add("threshold", threshold);
        parameters.add("background_id", backgroundId);
        parameters.add("nms_threshold", nmsThreashold);
        parameters.add("force_suppress", forceSuppress);
        parameters.add("nms_topk", nmsTopK);
        return getManager().invoke("MultiBoxDetection", inputs, parameters);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray getArray() {
        return array;
    }

    private MxNDManager getManager() {
        return array.getManager();
    }

    private int getGlobalPoolingDim() {
        int poolDim = getArray().getShape().dimension() - 2;
        if (poolDim < 1 || poolDim > 3) {
            throw new IllegalStateException(
                    "GlobalPooling only support"
                            + "1 to 3 Dimensions, "
                            + poolDim
                            + "D is not supported.");
        }
        return poolDim;
    }

    private Shape getGlobalPoolingShapes(long fillValue) {
        // determine pooling dimension according to input
        // input dimension minus 2 (batch and channel dim)
        int poolDim = getGlobalPoolingDim();
        long[] shape = new long[poolDim];
        Arrays.fill(shape, fillValue);
        return new Shape(shape);
    }
}
