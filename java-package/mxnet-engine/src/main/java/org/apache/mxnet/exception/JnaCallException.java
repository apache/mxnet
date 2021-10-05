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

package org.apache.mxnet.exception;

/** Thrown to indicate JNA functions are not called as expected. */
public class JnaCallException extends BaseException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new exception with the specified detail message. The cause is not initialized,
     * and may subsequently be initialized by a call to {@link #initCause}.
     *
     * @param message the detail message. The detail message is saved for later retrieval by the
     *     {@link #getMessage()} method.
     */
    public JnaCallException(String message) {
        super(message);
    }

    /**
     * \ Constructs a new exception with the specified detail message and cause.
     *
     * <p>Note that the detail message associated with {@code cause} is <i>not</i> automatically
     * incorporated in this exception's detail message.
     *
     * @param message the detail message that is saved for later retrieval by the {@link
     *     #getMessage()} method
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public JnaCallException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new exception with the specified cause and a detail message of {@code
     * (cause==null ? null : cause.toString())} which typically contains the class and detail
     * message of {@code cause}. This constructor is useful for exceptions that are little more than
     * wrappers for other throwables. For example, {@link java.security.PrivilegedActionException}.
     *
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public JnaCallException(Throwable cause) {
        super(cause);
    }
}
