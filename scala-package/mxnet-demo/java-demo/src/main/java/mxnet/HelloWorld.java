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
package mxnet;

import org.apache.mxnet.javaapi.*;
import java.util.Arrays;

public class HelloWorld {
    static NDArray$ NDArray = NDArray$.MODULE$;

    public static void main(String[] args) {
    	System.out.println("Hello World!");
        NDArray nd = new NDArray(new float[]{2.0f, 3.0f}, new Shape(new int[]{1, 2}), Context.cpu());
        System.out.println(nd.shape());
        NDArray nd2 = NDArray.dot(new dotParam(nd, nd.T()))[0];
        System.out.println(Arrays.toString(nd2.toArray()));
    }
}
