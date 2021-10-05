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

/** An enum that indicates whether gradient is required. */
public enum GradReq {
    NULL("null", 0),
    WRITE("write", 1),
    ADD("add", 3);

    private String type;
    private int value;

    GradReq(String type, int value) {
        this.type = type;
        this.value = value;
    }

    /**
     * Gets the type of this {@code GradReq}.
     *
     * @return the type
     */
    public String getType() {
        return type;
    }

    /**
     * Gets the value of this {@code GradType}.
     *
     * @return the value
     */
    public int getValue() {
        return value;
    }
}
