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

import com.sun.jna.Pointer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.util.PairList;
import org.apache.mxnet.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code Symbol} is an internal helper for symbolic model graphs used by the {@link
 * org.apache.mxnet.nn.SymbolBlock}.
 *
 * @see org.apache.mxnet.nn.SymbolBlock
 */
public class Symbol extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(Symbol.class);

    private String[] outputs;

    protected Symbol(MxResource parent, Pointer handle) {
        super(parent, handle);
    }

    static Symbol loadFromFile(MxResource parent, String path) {
        Pointer p = JnaUtils.createSymbolFromFile(path);
        return new Symbol(parent, p);
    }

    public static Symbol loadSymbol(MxResource parent, Path path) {
        return loadFromFile(parent, path.toAbsolutePath().toString());
    }

    /**
     * Loads a symbol from a json string.
     *
     * @param json the json string of the symbol.
     * @return the new symbol
     */
    public static Symbol loadJson(MxResource parent, String json) {
        Pointer pointer = JnaUtils.createSymbolFromString(json);
        return new Symbol(parent, pointer);
    }

    public String[] getOutputNames() {
        if (this.outputs == null) {
            this.outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return this.outputs;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free Symbol instance: %S", this.toJsonString()));
            super.freeSubResources();
            Pointer pointer = handle.getAndSet(null);
            if (pointer != null) {
                JnaUtils.freeSymbol(pointer);
            }
            setClosed();
            logger.debug(String.format("Finish to free Symbol instance: %S", this.toJsonString()));
        }
    }

    /**
     * Returns the output symbol by index.
     *
     * @param index the index of the output
     * @return the symbol output as a new symbol
     */
    public Symbol get(int index) {
        Pointer pointer = JnaUtils.getSymbolOutput(getInternals().getHandle(), index);
        return new Symbol(getParent(), pointer);
    }

    /**
     * Returns the output symbol with the given name.
     *
     * @param name the name of the symbol to return
     * @return the output symbol
     * @throws IllegalArgumentException Thrown if no output matches the name
     */
    public Symbol get(String name) {
        String[] out = getInternalOutputNames();
        int index = Utils.indexOf(out, name);
        if (index < 0) {
            throw new IllegalArgumentException("Cannot find output that matches name: " + name);
        }
        return get(index);
    }

    /**
     * Returns the symbol argument names.
     *
     * @return the symbol argument names
     */
    public String[] getArgNames() {
        return JnaUtils.listSymbolArguments(getHandle());
    }

    /**
     * Returns the MXNet auxiliary states for the symbol.
     *
     * @return the MXNet auxiliary states for the symbol
     */
    public String[] getAuxNames() {
        return JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    /**
     * Returns the symbol names.
     *
     * @return the symbol names
     */
    public String[] getAllNames() {
        return JnaUtils.listSymbolNames(getHandle());
    }

    /**
     * Returns the list of names for all internal outputs.
     *
     * @return a list of names
     */
    public List<String> getLayerNames() {
        String[] outputNames = getInternalOutputNames();
        String[] allNames = getAllNames();
        Set<String> allNamesSet = new LinkedHashSet<>(Arrays.asList(allNames));
        // Kill all params field and keep the output layer
        return Arrays.stream(outputNames)
                .filter(n -> !allNamesSet.contains(n))
                .collect(Collectors.toList());
    }

    private String[] getInternalOutputNames() {
        return JnaUtils.listSymbolOutputs(getInternals().getHandle());
    }

    /**
     * Returns the symbol internals.
     *
     * @return the symbol internals symbol
     */
    public Symbol getInternals() {
        Pointer pointer = JnaUtils.getSymbolInternals(getHandle());
        return new Symbol(getParent(), pointer);
    }

    /**
     * Infers the shapes for all parameters inside a symbol from the given input shapes.
     *
     * @param pairs the given input name and shape
     * @return a map of arguments with names and shapes
     */
    public Map<String, Shape> inferShape(PairList<String, Shape> pairs) {
        List<List<Shape>> shapes = JnaUtils.inferShape(this, pairs);
        if (shapes == null) {
            throw new IllegalArgumentException("Cannot infer shape based on the data provided!");
        }
        List<Shape> argShapes = shapes.get(0);
        List<Shape> outputShapes = shapes.get(1);
        List<Shape> auxShapes = shapes.get(2);
        // TODO: add output to the map
        String[] argNames = getArgNames();
        String[] auxNames = getAuxNames();
        String[] outputNames = getOutputNames();
        Map<String, Shape> shapesMap = new ConcurrentHashMap<>();
        for (int i = 0; i < argNames.length; i++) {
            shapesMap.put(argNames[i], argShapes.get(i));
        }
        for (int i = 0; i < auxNames.length; i++) {
            shapesMap.put(auxNames[i], auxShapes.get(i));
        }
        for (int i = 0; i < outputNames.length; i++) {
            shapesMap.put(outputNames[i], outputShapes.get(i));
        }
        return shapesMap;
    }

    /**
     * [Experimental] Add customized optimization on the Symbol.
     *
     * <p>This method can be used with EIA or TensorRT for model acceleration
     *
     * @param backend backend name
     * @param device the device assigned
     * @return optimized Symbol
     */
    public Symbol optimizeFor(String backend, Device device) {
        return new Symbol(getParent(), JnaUtils.optimizeFor(this, backend, device));
    }

    /**
     * Converts Symbol to json string for saving purpose.
     *
     * @return the json string
     */
    public String toJsonString() {
        return JnaUtils.getSymbolString(getHandle());
    }
}
