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

package org.apache.mxnet.jna;

import com.sun.jna.Pointer;
import java.util.List;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.types.SparseFormat;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A FunctionInfo represents an operator (ie function) within the MXNet Engine. */
public class FunctionInfo {

    private Pointer handle;
    private String name;
    private PairList<String, String> arguments;

    private static final Logger logger = LoggerFactory.getLogger(FunctionInfo.class);

    FunctionInfo(Pointer pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    /**
     * Returns the name of the operator.
     *
     * @return the name of the operator
     */
    public String getFunctionName() {
        return name;
    }

    /**
     * Returns the names of the params to the operator.
     *
     * @return the names of the params to the operator
     */
    public List<String> getArgumentNames() {
        return arguments.keys();
    }

    /**
     * Returns the types of the operator arguments.
     *
     * @return the types of the operator arguments
     */
    public List<String> getArgumentTypes() {
        return arguments.values();
    }
    /**
     * Calls an operator with the given arguments.
     *
     * @param src the input NDArray(s) to the operator
     * @param dest the destination NDArray(s) to be overwritten with the result of the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     * @return the error code or zero for no errors
     */
    public int invoke(NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        checkDevices(src);
        checkDevices(dest);
        return JnaUtils.imperativeInvoke(handle, src, dest, params).size();
    }

    /**
     * Calls an operator with the given arguments.
     *
     * @param parent {@link MxResource} for the current instance
     * @param src the input NDArray(s) to the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     * @return the error code or zero for no errors
     */
    public NDArray[] invoke(MxResource parent, NDArray[] src, PairList<String, ?> params) {
        checkDevices(src);
        PairList<Pointer, SparseFormat> pairList =
                JnaUtils.imperativeInvoke(handle, src, null, params);
        return pairList.stream()
                .map(
                        pair -> {
                            if (pair.getValue() != SparseFormat.DENSE) {
                                return NDArray.create(parent, pair.getKey(), pair.getValue());
                            }
                            return NDArray.create(parent, pair.getKey());
                        })
                .toArray(NDArray[]::new);
    }

    private void checkDevices(NDArray[] src) {
        // check if all the NDArrays are in the same device
        if (logger.isDebugEnabled() && src.length > 1) {
            Device device = src[0].getDevice();
            for (int i = 1; i < src.length; ++i) {
                if (!device.equals(src[i].getDevice())) {
                    logger.warn(
                            "Please make sure all the NDArrays are in the same device. You can call toDevice() to move the NDArray to the desired Device.");
                }
            }
        }
    }
}
