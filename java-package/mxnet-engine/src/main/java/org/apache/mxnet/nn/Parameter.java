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

package org.apache.mxnet.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.UUID;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.exception.MalformedModelException;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDSerializer;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code Parameter} is a container class that holds a learnable parameter of a model.
 *
 * <p>Every {@code Parameter} is associated with a {@link SymbolBlock}. The output of the block's
 * forward function depends on the values in the {@code Parameter}. During training, the values in
 * the {@code Parameter} are updated to reflect the training data. This process forms the crux of
 * learning.
 *
 * @see <a href="https://d2l.djl.ai/chapter_deep-learning-computation/parameters.html">The D2L
 *     chapter on parameter management</a>
 */
public class Parameter extends MxResource {
    private static final Logger logger = LoggerFactory.getLogger(Parameter.class);

    private static final byte VERSION = 1;

    private String id;
    private String name;
    private Shape shape;
    private Type type;
    private NDArray array;
    private boolean requiresGrad;

    Parameter(Builder builder) {
        this.id = UUID.randomUUID().toString();
        this.name = builder.name;
        this.shape = builder.shape;
        this.type = builder.type;
        this.array = builder.array;
        this.requiresGrad = builder.requiresGrad;
    }

    /**
     * Gets the ID of this {@code Parameter}.
     *
     * @return the ID of this {@code Parameter}
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the name of this {@code Parameter}.
     *
     * @return the name of this {@code Parameter}
     */
    public String getName() {
        return name == null ? "" : name;
    }

    /**
     * Gets the type of this {@code Parameter}.
     *
     * @return the type of this {@code Parameter}
     */
    public Type getType() {
        return type;
    }

    /**
     * Sets the values of this {@code Parameter}.
     *
     * @param array the {@link NDArray} that contains values of this {@code Parameter}
     */
    public void setArray(NDArray array) {
        if (shape != null) {
            throw new IllegalStateException("array has been set! Use either setArray or setShape");
        }
        this.array = array;
        shape = array.getShape();
        array.setName(name);
    }

    /**
     * Sets the shape of this {@code Parameter}.
     *
     * @param shape the shape of this {@code Parameter}
     */
    public void setShape(Shape shape) {
        if (array != null) {
            throw new IllegalStateException("array has been set! Use either setArray or setShape");
        }
        this.shape = shape;
    }

    /**
     * Gets the values of this {@code Parameter} as an {@link NDArray}.
     *
     * @return an {@link NDArray} that contains values of this {@code Parameter}
     */
    public NDArray getArray() {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return array;
    }

    /**
     * Returns whether this parameter needs gradients to be computed.
     *
     * @return whether this parameter needs gradients to be computed
     */
    public boolean requiresGradient() {
        return requiresGrad;
    }

    /**
     * Checks if this {@code Parameter} is initialized.
     *
     * @return {@code true} if this {@code Parameter} is initialized
     */
    public boolean isInitialized() {
        return array != null;
    }

    /**
     * Initializes the parameter, with given {@link DataType} for the given expected input shapes.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param dataType the datatype of the {@code Parameter}
     */
    public void initialize(MxResource parent, DataType dataType, Device device) {
        Objects.requireNonNull(shape, "No parameter shape has been set");
        if (requiresGradient()) {
            array.setRequiresGradient(true);
        }
    }

    /**
     * Writes the parameter NDArrays to the given output stream.
     *
     * @param dos the output stream to write to
     * @throws IOException if the write operation fails
     */
    public void save(DataOutputStream dos) throws IOException {
        if (!isInitialized()) {
            dos.writeChar('N');
            return;
        }

        dos.writeChar('P');
        dos.writeByte(VERSION);
        dos.writeUTF(getName());
        dos.write(array.encode());
    }

    /**
     * Loads parameter NDArrays from InputStream.
     *
     * <p>Currently, we cannot deserialize into the exact subclass of NDArray. The SparseNDArray
     * will be loaded as NDArray only.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param dis the InputStream
     * @throws IOException if failed to read
     * @throws MalformedModelException Exception thrown when model is not in expected format
     *     (parameters).
     */
    public void load(MxResource parent, DataInputStream dis)
            throws IOException, MalformedModelException {
        char magic = dis.readChar();
        if (magic == 'N') {
            return;
        } else if (magic != 'P') {
            throw new MalformedModelException("Invalid input data.");
        }

        // Version
        byte version = dis.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }

        String parameterName = dis.readUTF();
        if (!parameterName.equals(getName())) {
            throw new MalformedModelException(
                    "Unexpected parameter name: " + parameterName + ", expected: " + name);
        }

        array = NDSerializer.decode(parent, dis);
        // set the shape of the parameter and prepare() can be skipped
        shape = array.getShape();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free Symbol instance: %S", this.getUid()));
            super.freeSubResources();
            if (array != null) {
                array.close();
                array = null;
            }
            setClosed();
            logger.debug(String.format("Start to free Symbol instance: %S", this.getUid()));
        }
    }

    /**
     * Creates a builder to build a {@code Parameter}.
     *
     * <p>The methods start with {@code set} are required fields, and {@code opt} for optional
     * fields.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Enumerates the types of {@link Parameter}. */
    public enum Type {
        WEIGHT,
        BIAS,
        GAMMA,
        BETA,
        RUNNING_MEAN,
        RUNNING_VAR,
        OTHER;
    }

    /** A Builder to construct a {@code Parameter}. */
    public static final class Builder {
        String name;
        Shape shape;
        Type type;
        NDArray array;
        boolean requiresGrad = true;

        /**
         * Sets the name of the {@code Parameter}.
         *
         * @param name the name of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder setName(String name) {
            this.name = name;
            return this;
        }

        /**
         * Sets the {@code Type} of the {@code Parameter}.
         *
         * @param type the {@code Type} of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder setType(Type type) {
            this.type = type;
            return this;
        }

        /**
         * Sets the shape of the {@code Parameter}.
         *
         * @param shape the shape of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder optShape(Shape shape) {
            this.shape = shape;
            return this;
        }

        /**
         * Sets the array of the {@code Parameter}.
         *
         * @param array the array of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder optArray(NDArray array) {
            this.array = array;
            return this;
        }

        /**
         * Sets if the {@code Parameter} requires gradient.
         *
         * @param requiresGrad if the {@code Parameter} requires gradient
         * @return this {@code Parameter}
         */
        public Builder optRequiresGrad(boolean requiresGrad) {
            this.requiresGrad = requiresGrad;
            return this;
        }

        /**
         * Builds a {@code Parameter} instance.
         *
         * @return the {@code Parameter} instance
         */
        public Parameter build() {
            return new Parameter(this);
        }
    }
}
