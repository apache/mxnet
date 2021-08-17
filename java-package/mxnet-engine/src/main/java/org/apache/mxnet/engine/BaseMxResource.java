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

import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The top-level {@link MxResource} instance, with no parent Resource to manage. The {@link
 * BaseMxResource} instance will be lazy loaded when the first time called, like when {@link Model}
 * instance is loaded for the first time.
 */
public final class BaseMxResource extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(BaseMxResource.class);

    private static BaseMxResource systemMxResource;

    protected BaseMxResource() {
        super();
        // Workaround MXNet engine lazy initialization issue
        JnaUtils.getAllOpNames();

        JnaUtils.setNumpyMode(JnaUtils.NumpyMode.GLOBAL_ON);

        // Workaround MXNet shutdown crash issue
        Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD
    }

    /**
     * Getter method for the singleton {@code systemMxResource} instance.
     *
     * @return The top-leve {@link BaseMxResource} instance.
     */
    public static synchronized BaseMxResource getSystemMxResource() {
        if (systemMxResource == null) {
            systemMxResource = new BaseMxResource();
        }
        return systemMxResource;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free BaseMxResource instance: %S", this.getUid()));
            // only clean sub resources
            JnaUtils.waitAll();
            super.freeSubResources();
            setClosed(true);
            logger.debug(
                    String.format("Finish to free BaseMxResource instance: %S", this.getUid()));
        }
    }
}
