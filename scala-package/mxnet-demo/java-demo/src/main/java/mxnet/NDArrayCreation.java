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

public class NDArrayCreation {
    static NDArray$ NDArray = NDArray$.MODULE$;
    public static void main(String[] args) {

        // Create new NDArray
        NDArray nd = new NDArray(new float[]{2.0f, 3.0f}, new Shape(new int[]{1, 2}), Context.cpu());
        System.out.println(nd);

        // create new Double NDArray
        NDArray ndDouble = new NDArray(new double[]{2.0d, 3.0d}, new Shape(new int[]{2, 1}), Context.cpu());
        System.out.println(ndDouble);

        // create ones
        NDArray ones = NDArray.ones(Context.cpu(), new int[] {1, 2, 3});
        System.out.println(ones);

        // random
        NDArray random = NDArray.random_uniform(
                new random_uniformParam()
                        .setLow(0.0f)
                        .setHigh(2.0f)
                        .setShape(new Shape(new int[]{10, 10}))
        )[0];
        System.out.println(random);
    }
}
