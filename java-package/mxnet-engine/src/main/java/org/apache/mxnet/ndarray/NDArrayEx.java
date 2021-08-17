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

package org.apache.mxnet.ndarray;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.OpParams;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;

/** An internal interface that encapsulates engine specific operations. */
@SuppressWarnings("MissingJavadocMethod")
public class NDArrayEx {

    private static final NDArrayIndexer INDEXER = new NDArrayIndexer();

    private NDArray array;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    NDArrayEx(NDArray parent) {
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
    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n the Value to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    public NDArray rdiv(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return NDArray.invoke(getArray().getParent(), "_rdiv_scalar", array, params);
    }

    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    public NDArray rdiv(NDArray b) {
        return b.div(array);
    }

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    public NDArray rdivi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        NDArray.invoke("_rdiv_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    public NDArray rdivi(NDArray b) {
        NDArray.invoke("elemwise_div", new NDArray[] {b, array}, new NDArray[] {array}, null);
        return array;
    }

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return a copy of array after reverse subtraction
     */
    public NDArray rsub(Number n) {
        return array.sub(n).neg();
    }

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return a copy of the array after reverse subtraction
     */
    public NDArray rsub(NDArray b) {
        return array.sub(b).neg();
    }

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    public NDArray rsubi(Number n) {
        return array.subi(n).negi();
    }

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    public NDArray rsubi(NDArray b) {
        return array.subi(b).negi();
    }

    public NDArray rmod(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return NDArray.invoke(getArray().getParent(), "_npi_rmod_scalar", array, params);
    }

    /**
     * Applies reverse remainder of division with a scalar.
     *
     * @param b the value to use for reverse division
     * @return a copy of array after applying reverse division
     */
    public NDArray rmod(NDArray b) {
        return b.mod(array);
    }

    /**
     * Applies in place reverse remainder of division with a scalar.
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    public NDArray rmodi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        NDArray.invoke("_npi_rmod_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    /**
     * Applies in place reverse remainder of division.
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    public NDArray rmodi(NDArray b) {
        NDArray.invoke("_npi_mod", new NDArray[] {b, array}, new NDArray[] {array}, null);
        return array;
    }

    /**
     * Reverses the power of each element being raised in the {@code NDArray}.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    public NDArray rpow(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return NDArray.invoke(getArray().getParent(), "_npi_rpower_scalar", array, params);
    }

    /**
     * Reverses the power of each element being raised in the {@code NDArray} in place.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    public NDArray rpowi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        NDArray.invoke("_npi_rpower_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
        return array;
    }

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////
    /**
     * Computes rectified linear activation.
     *
     * @return a copy of array after applying relu
     */
    public NDArray relu() {
        OpParams params = new OpParams();
        params.addParam("act_type", "relu");
        return NDArray.invoke(getArray().getParent(), "_npx_activation", array, params);
    }

    public NDArray sigmoid() {
        OpParams params = new OpParams();
        params.addParam("act_type", "sigmoid");
        return NDArray.invoke(getArray().getParent(), "_npx_activation", array, params);
    }

    public NDArray tanh() {
        OpParams params = new OpParams();
        params.addParam("act_type", "tanh");
        return NDArray.invoke(getArray().getParent(), "_npx_activation", array, params);
    }

    public NDArray softPlus() {
        OpParams params = new OpParams();
        params.addParam("act_type", "softrelu");
        return NDArray.invoke(getArray().getParent(), "_npx_activation", array, params);
    }

    public NDArray softSign() {
        OpParams params = new OpParams();
        params.addParam("act_type", "softsign");
        return NDArray.invoke(getArray().getParent(), "_npx_activation", array, params);
    }

    public NDArray leakyRelu(float alpha) {
        OpParams params = new OpParams();
        params.addParam("act_type", "leaky");
        params.addParam("slope", alpha);
        return NDArray.invoke(getArray().getParent(), "_npx_leaky_relu", array, params);
    }

    public NDArray elu(float alpha) {
        OpParams params = new OpParams();
        params.addParam("act_type", "elu");
        params.addParam("slope", alpha);
        return NDArray.invoke(getArray().getParent(), "_npx_leaky_relu", array, params);
    }

    public NDArray selu() {
        OpParams params = new OpParams();
        params.addParam("act_type", "selu");
        return NDArray.invoke(getArray().getParent(), "_npx_leaky_relu", array, params);
    }

    public NDArray gelu() {
        OpParams params = new OpParams();
        params.addParam("act_type", "gelu");
        return NDArray.invoke(getArray().getParent(), "_npx_leaky_relu", array, params);
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    public NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        OpParams params = new OpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "max");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        return NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params);
    }

    public NDArray globalMaxPool() {
        OpParams params = new OpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "max");
        params.addParam("global_pool", true);
        try (NDArray temp =
                NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    public NDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        OpParams params = new OpParams();
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "avg");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");
        params.addParam("count_include_pad", countIncludePad);
        return NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params);
    }

    public NDArray globalAvgPool() {
        OpParams params = new OpParams();
        params.add("kernel", getGlobalPoolingShapes(1));
        params.add("pad", getGlobalPoolingShapes(0));
        params.add("pool_type", "avg");
        params.addParam("global_pool", true);
        try (NDArray temp =
                NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    public NDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer instead");
        }
        OpParams params = new OpParams();
        params.addParam("p_value", (int) normType);
        params.addParam("kernel", kernelShape);
        params.add("pool_type", "lp");
        params.addParam("stride", stride);
        params.addParam("pad", padding);
        params.add("pooling_convention", ceilMode ? "full" : "valid");

        return NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params);
    }

    public NDArray globalLpPool(float normType) {
        if (((int) normType) != normType) {
            throw new IllegalArgumentException(
                    "float type of normType is not supported in MXNet engine, please use integer instead");
        }
        OpParams params = new OpParams();
        params.add("pool_type", "lp");
        params.addParam("p_value", (int) normType);
        params.addParam("global_pool", true);
        try (NDArray temp =
                NDArray.invoke(getArray().getParent(), "_npx_pooling", getArray(), params)) {
            return temp.reshape(temp.getShape().size(0), temp.getShape().size(1));
        }
    }

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

    //    public void adadeltaUpdate(
    //            MxNDList inputs,
    //            MxNDList weights,
    //            float weightDecay,
    //            float rescaleGrad,
    //            float clipGrad,
    //            float rho,
    //            float epsilon) {
    //        MxNDArray weight = inputs.get(0);
    //        MxNDArray grad = inputs.get(1);
    //        MxNDArray s = inputs.get(2);
    //        MxNDArray delta = inputs.get(3);
    //
    //        // create a baseManager to close all intermediate MxNDArrays
    //        try (NDManager subManager = NDManager.newBaseManager()) {
    //            subManager.tempAttachAll(inputs, weights);
    //
    //            // Preprocess Gradient
    //            grad.muli(rescaleGrad);
    //            if (clipGrad > 0) {
    //                grad = grad.clip(-clipGrad, clipGrad);
    //            }
    //            grad.addi(weight.mul(weightDecay));
    //
    //            // Update s, g, and delta
    //            s.muli(rho).addi(grad.square().mul(1 - rho));
    //            MxNDArray g = delta.add(epsilon).sqrt().div(s.add(epsilon).sqrt()).mul(grad);
    //            delta.muli(rho).addi(g.square().mul(1 - rho));
    //
    //            // Update weight
    //            weight.subi(g);
    //        }
    //    }

    public void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon) {
        OpParams params = new OpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("epsilon", epsilon);

        NDArray.invoke("adagrad_update", inputs, weights, params);
    }

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
        OpParams params = new OpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("beta1", beta1);
        params.addParam("beta2", beta2);
        params.addParam("epsilon", epsilon);
        params.addParam("lazy_update", lazyUpdate);

        NDArray.invoke("adam_update", inputs, weights, params);
    }

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
        OpParams params = new OpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);

        params.addParam("gamma1", gamma1);
        params.addParam("epsilon", epsilon);

        if (!centered) {
            NDArray.invoke("rmsprop_update", inputs, weights, params);
        } else {
            params.addParam("gamma2", gamma2);

            NDArray.invoke("rmspropalex_update", inputs, weights, params);
        }
    }

    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        OpParams params = new OpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("momentum", momentum);
        NDArray.invoke("nag_mom_update", inputs, weights, params);
    }

    public void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate) {
        OpParams params = new OpParams();
        params.addParam("lr", learningRate);
        params.addParam("wd", weightDecay);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGrad);
        params.addParam("lazy_update", lazyUpdate);

        if (momentum != 0) {
            params.addParam("momentum", momentum);
            NDArray.invoke("sgd_mom_update", inputs, weights, params);
        } else {
            NDArray.invoke("sgd_update", inputs, weights, params);
        }
    }

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    public NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        OpParams params = new OpParams();
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

        return NDArray.invoke(getArray().getParent(), "_npx_convolution", inputs, params);
    }

    public NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        OpParams params = new OpParams();
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

        return NDArray.invoke(getArray().getParent(), "_npx_deconvolution", inputs, params);
    }

    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        OpParams params = new OpParams();
        params.addParam("num_hidden", weight.size(0));
        params.addParam("flatten", false);
        params.addParam("no_bias", bias == null);
        NDList inputs = new NDList(input, weight);
        if (bias != null) {
            inputs.add(bias);
        }

        return NDArray.invoke(getArray().getParent(), "_npx_fully_connected", inputs, params);
    }

    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
        if (!sparse.equals(SparseFormat.DENSE) && !sparse.equals(SparseFormat.ROW_SPARSE)) {
            throw new IllegalArgumentException("MXNet only supports row sparse");
        }
        OpParams params = new OpParams();
        long inputDim = weight.getShape().get(0);
        long outputDim = weight.getShape().get(1);
        params.addParam("input_dim", inputDim);
        params.addParam("output_dim", outputDim);
        params.addParam("sparse_grad", sparse.getValue());
        return NDArray.invoke(
                getArray().getParent(), "_npx_embedding", new NDList(input, weight), params);
    }

    public NDList prelu(NDArray input, NDArray alpha) {
        OpParams params = new OpParams();
        params.addParam("act_type", "prelu");
        return NDArray.invoke(
                getArray().getParent(), "_npx_leaky_relu", new NDList(input, alpha), params);
    }

    public NDList dropout(NDArray input, float rate, boolean training) {
        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of dropout in MXNet should align with the mode of GradientCollector");
        }

        OpParams params = new OpParams();
        params.addParam("p", rate);

        return NDArray.invoke(getArray().getParent(), "_npx_dropout", new NDList(input), params);
    }

    public NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        params.addParam("fix_gamma", gamma == null);
        params.addParam("eps", eps);
        params.addParam("momentum", momentum);

        if (training != JnaUtils.autogradIsTraining()) {
            throw new IllegalArgumentException(
                    "the mode of batchNorm in MXNet should align with the mode of GradientCollector");
        }

        return NDArray.invoke(
                getArray().getParent(),
                "_npx_batch_norm",
                new NDList(input, gamma, beta, runningMean, runningVar),
                params);
    }

    //    public MxNDList rnn(
    //            MxNDArray input,
    //            MxNDArray state,
    //            MxNDList params,
    //            boolean hasBiases,
    //            int numLayers,
    //            RNN.Activation activation,
    //            double dropRate,
    //            boolean training,
    //            boolean bidirectional,
    //            boolean batchFirst) {
    //        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
    //        Preconditions.checkArgument(
    //                params.size() == numParams,
    //                "The size of Params is incorrect expect "
    //                        + numParams
    //                        + " parameters but got "
    //                        + params.size());
    //
    //        if (training != JnaUtils.autogradIsTraining()) {
    //            throw new IllegalArgumentException(
    //                    "the mode of rnn in MXNet should align with the mode of
    // GradientCollector");
    //        }
    //
    //        if (batchFirst) {
    //            input = input.swapAxes(0, 1);
    //        }
    //
    //        MxOpParams opParams = new MxOpParams();
    //        opParams.addParam("p", dropRate);
    //        opParams.addParam("state_size", state.getShape().tail());
    //        opParams.addParam("num_layers", numLayers);
    //        opParams.addParam("bidirectional", bidirectional);
    //        opParams.addParam("state_outputs", true);
    //        opParams.addParam("mode", activation == RNN.Activation.TANH ? "rnn_tanh" :
    // "rnn_relu");
    //
    //        MxNDList inputs = new MxNDList();
    //        inputs.add(input);
    //
    //        try (MxNDList temp = new MxNDList()) {
    //            for (MxNDArray param : params) {
    //                temp.add(param.flatten());
    //            }
    //            MxNDArray tempParam = MxNDArrays.concat(temp);
    //            tempParam.attach(input.getManager());
    //            inputs.add(tempParam);
    //        }
    //
    //        inputs.add(state);
    //
    //        if (!batchFirst) {
    //            return getManager().invoke("_npx_rnn", inputs, opParams);
    //        }
    //
    //        MxNDList result = getManager().invoke("_npx_rnn", inputs, opParams);
    //        try (MxNDArray temp = result.head()) {
    //            return new MxNDList(temp.swapAxes(0, 1), result.get(1));
    //        }
    //    }

    //    public MxNDList gru(
    //            MxNDArray input,
    //            MxNDArray state,
    //            MxNDList params,
    //            boolean hasBiases,
    //            int numLayers,
    //            double dropRate,
    //            boolean training,
    //            boolean bidirectional,
    //            boolean batchFirst) {
    //        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
    //        Preconditions.checkArgument(
    //                params.size() == numParams,
    //                "The size of Params is incorrect expect "
    //                        + numParams
    //                        + " parameters but got "
    //                        + params.size());
    //
    //        if (training != JnaUtils.autogradIsTraining()) {
    //            throw new IllegalArgumentException(
    //                    "the mode of gru in MXNet should align with the mode of
    // GradientCollector");
    //        }
    //
    //        if (batchFirst) {
    //            input = input.swapAxes(0, 1);
    //        }
    //
    //        MxOpParams opParams = new MxOpParams();
    //        opParams.addParam("p", dropRate);
    //        opParams.addParam("state_size", state.getShape().tail());
    //        opParams.addParam("num_layers", numLayers);
    //        opParams.addParam("bidirectional", bidirectional);
    //        opParams.addParam("state_outputs", true);
    //        opParams.addParam("mode", "gru");
    //
    //        MxNDList inputs = new MxNDList();
    //        inputs.add(input);
    //
    //        try (MxNDList temp = new MxNDList()) {
    //            for (MxNDArray param : params) {
    //                temp.add(param.flatten());
    //            }
    //            MxNDArray tempParam = MxNDArrays.concat(temp);
    //            tempParam.attach(input.getManager());
    //            inputs.add(tempParam);
    //        }
    //
    //        inputs.add(state);
    //
    //        if (!batchFirst) {
    //            return getManager().invoke("_npx_rnn", inputs, opParams);
    //        }
    //
    //        MxNDList result = getManager().invoke("_npx_rnn", inputs, opParams);
    //        try (MxNDArray temp = result.head()) {
    //            return new MxNDList(temp.swapAxes(0, 1), result.get(1));
    //        }
    //    }
    //
    //    public MxNDList lstm(
    //            MxNDArray input,
    //            MxNDList states,
    //            MxNDList params,
    //            boolean hasBiases,
    //            int numLayers,
    //            double dropRate,
    //            boolean training,
    //            boolean bidirectional,
    //            boolean batchFirst) {
    //        int numParams = numLayers * ((hasBiases) ? 4 : 2) * ((bidirectional) ? 2 : 1);
    //        Preconditions.checkArgument(
    //                params.size() == numParams,
    //                "The size of Params is incorrect expect "
    //                        + numParams
    //                        + " parameters but got "
    //                        + params.size());
    //
    //        if (training != JnaUtils.autogradIsTraining()) {
    //            throw new IllegalArgumentException(
    //                    "the mode of lstm in MXNet should align with the mode of
    // GradientCollector");
    //        }
    //
    //        if (batchFirst) {
    //            input = input.swapAxes(0, 1);
    //        }
    //
    //        MxOpParams opParams = new MxOpParams();
    //        opParams.addParam("mode", "lstm");
    //        opParams.addParam("p", dropRate);
    //        opParams.addParam("state_size", states.head().getShape().tail());
    //        opParams.addParam("state_outputs", true);
    //        opParams.addParam("num_layers", numLayers);
    //        opParams.addParam("bidirectional", bidirectional);
    //        opParams.addParam("lstm_state_clip_nan", true);
    //
    //        MxNDList inputs = new MxNDList();
    //        inputs.add(input);
    //        try (MxNDList temp = new MxNDList()) {
    //            for (MxNDArray param : params) {
    //                temp.add(param.flatten());
    //            }
    //            MxNDArray tempParam = MxNDArrays.concat(temp);
    //            tempParam.attach(input.getManager());
    //            inputs.add(tempParam);
    //        }
    //        inputs.addAll(states);
    //
    //        if (!batchFirst) {
    //            return getManager().invoke("_npx_rnn", inputs, opParams);
    //        }
    //
    //        MxNDList result = getManager().invoke("_npx_rnn", inputs, opParams);
    //        try (MxNDArray temp = result.head()) {
    //            return new MxNDList(temp.swapAxes(0, 1), result.get(1), result.get(2));
    //        }
    //    }

    ////////////////////////////////////////
    // Image and CV
    ////////////////////////////////////////

    public NDArray normalize(float[] mean, float[] std) {
        OpParams params = new OpParams();
        params.addTupleParam("mean", mean);
        params.addTupleParam("std", std);
        return NDArray.invoke(getArray().getParent(), "_npx__image_normalize", array, params);
    }

    public NDArray toTensor() {
        return NDArray.invoke(getArray().getParent(), "_npx__image_to_tensor", array, null);
    }

    public NDArray resize(int width, int height, int interpolation) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("attempt to resize of an empty MxNDArray");
        }
        OpParams params = new OpParams();
        params.addTupleParam("size", width, height);
        params.addParam("interp", interpolation);
        return NDArray.invoke(getArray().getParent(), "_npx__image_resize", array, params);
    }

    public NDArray crop(int x, int y, int width, int height) {
        OpParams params = new OpParams();
        params.add("x", x);
        params.add("y", y);
        params.add("width", width);
        params.add("height", height);
        return NDArray.invoke(getArray().getParent(), "_npx__image_crop", array, params);
    }

    public NDArray randomFlipLeftRight() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipLeftRight is not supported on GPU");
        }
        return NDArray.invoke(
                getArray().getParent(), "_npx__image_random_flip_left_right", array, null);
    }

    public NDArray randomFlipTopBottom() {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomFlipTopBottom is not supported on GPU");
        }
        return NDArray.invoke(
                getArray().getParent(), "_npx__image_random_flip_top_bottom", array, null);
    }

    public NDArray randomBrightness(float brightness) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomBrightness is not supported on GPU");
        }
        OpParams params = new OpParams();
        float min = Math.max(0, 1 - brightness);
        float max = 1 + brightness;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return NDArray.invoke(
                getArray().getParent(), "_npx__image_random_brightness", array, params);
    }

    public NDArray randomHue(float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomHue is not supported on GPU");
        }
        OpParams params = new OpParams();
        float min = Math.max(0, 1 - hue);
        float max = 1 + hue;
        params.addParam("min_factor", min);
        params.addParam("max_factor", max);
        return NDArray.invoke(getArray().getParent(), "_npx__image_random_hue", array, params);
    }

    public NDArray randomColorJitter(
            float brightness, float contrast, float saturation, float hue) {
        if (array.getDevice().getDeviceType().equals(Device.Type.GPU)) {
            throw new UnsupportedOperationException("randomColorJitter is not supported on GPU");
        }
        OpParams params = new OpParams();
        params.addParam("brightness", brightness);
        params.addParam("contrast", contrast);
        params.addParam("saturation", saturation);
        params.addParam("hue", hue);
        return NDArray.invoke(
                getArray().getParent(), "_npx__image_random_color_jitter", array, params);
    }

    public NDArrayIndexer getIndexer() {
        return INDEXER;
    }

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    @SuppressWarnings("PMD.UseTryWithResources")
    public NDArray where(NDArray condition, NDArray other) {
        NDArray array1;
        NDArray array2;
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
            return NDArray.invoke(
                    getArray().getParent(),
                    "where",
                    new NDArray[] {condition, array1, array2},
                    null);
        } finally {
            if (array1 != array) {
                array1.close();
            }
            if (array2 != other) {
                array2.close();
            }
        }
    }

    public NDArray stack(NDList arrays, int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
        return NDArray.invoke(getArray().getParent(), "_npi_stack", srcArray, params);
    }

    /**
     * Check two criteria of concat input: 1. no scalar 2. dimensions of all the array must be the
     * same.
     *
     * @param list input {@link NDList}
     */
    public static void checkConcatInput(NDList list) {
        NDArray[] arrays = list.toArray(new NDArray[0]);
        if (Stream.of(arrays).allMatch(array -> array.getShape().dimension() == 0)) {
            throw new IllegalArgumentException(
                    "scalar(zero-dimensional) arrays cannot be concatenated");
        }
        int dimension = arrays[0].getShape().dimension();
        for (int i = 1; i < arrays.length; i++) {
            if (arrays[i].getShape().dimension() != dimension) {
                throw new IllegalArgumentException(
                        "all the input arrays must have same number of dimensions, but the array at index 0 has "
                                + dimension
                                + " dimension(s) and the array at index "
                                + i
                                + " has "
                                + arrays[i].getShape().dimension()
                                + " dimension(s)");
            }
        }
    }

    public NDArray concat(NDList list, int axis) {
        checkConcatInput(list);

        OpParams params = new OpParams();
        // MXNet backend use dim as argument name
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[list.size() + 1];
        srcArray[0] = array;
        System.arraycopy(list.toArray(new NDArray[0]), 0, srcArray, 1, list.size());
        return NDArray.invoke(getArray().getParent(), "_npi_concatenate", srcArray, params);
    }

    public NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples) {
        OpParams parameters = new OpParams();
        parameters.add("minimum_negative_samples", minNegativeSamples);
        parameters.add("overlap_threshold", iouThreshold);
        parameters.add("ignore_label", ignoreLabel);
        parameters.add("negative_mining_ratio", negativeMiningRatio);
        parameters.add("negative_mining_thresh", negativeMiningThreshold);
        return NDArray.invoke(getArray().getParent(), "MultiBoxTarget", inputs, parameters);
    }

    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        OpParams parameters = new OpParams();
        parameters.add("sizes", sizes);
        parameters.add("ratios", ratios);
        parameters.add("steps", steps);
        parameters.add("offsets", offsets);
        parameters.add("clip", clip);
        return NDArray.invoke(
                getArray().getParent(), "MultiBoxPrior", new NDList(array), parameters);
    }

    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreashold,
            boolean forceSuppress,
            int nmsTopK) {
        OpParams parameters = new OpParams();
        parameters.add("clip", clip);
        parameters.add("threshold", threshold);
        parameters.add("background_id", backgroundId);
        parameters.add("nms_threshold", nmsThreashold);
        parameters.add("force_suppress", forceSuppress);
        parameters.add("nms_topk", nmsTopK);
        return NDArray.invoke(getArray().getParent(), "MultiBoxDetection", inputs, parameters);
    }

    public NDArray getArray() {
        return array;
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
