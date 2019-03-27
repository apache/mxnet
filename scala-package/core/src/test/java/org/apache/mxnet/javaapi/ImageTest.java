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

package org.apache.mxnet.javaapi;

import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;

public class ImageTest {

    private static String imLocation;

    private static void downloadUrl(String url, String filePath, int maxRetry) throws Exception{
        File tmpFile = new File(filePath);
        Boolean success = false;
        if (!tmpFile.exists()) {
            while (maxRetry > 0 && !success) {
                try {
                    FileUtils.copyURLToFile(new URL(url), tmpFile);
                    success = true;
                } catch(Exception e){
                   maxRetry -= 1;
                }
            }
        } else {
            success = true;
        }
        if (!success) throw new Exception("$url Download failed!");
    }

    @BeforeClass
    public static void downloadFile() throws Exception {
        String tempDirPath = System.getProperty("java.io.tmpdir");
        imLocation = tempDirPath + "/inputImages/Pug-Cookie.jpg";
        downloadUrl("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg",
                imLocation, 3);
    }

    @Test
    public void testImageProcess() throws Exception {
        NDArray nd = Image.imRead(imLocation, 1, true);
        assertArrayEquals(nd.shape().toArray(), new int[]{576, 1024, 3});
        NDArray nd2 = Image.imResize(nd, 224, 224, null);
        assertArrayEquals(nd2.shape().toArray(), new int[]{224, 224, 3});
        NDArray cropped = Image.fixedCrop(nd, 0, 0, 224, 224);
        Image.toImage(cropped);
        BufferedImage buf = ImageIO.read(new File(imLocation));
        Map<String, Integer> map = new HashMap<>();
        map.put("xmin", 190);
        map.put("xmax", 850);
        map.put("ymin", 50);
        map.put("ymax", 450);
        List<Map<String, Integer>> box = new ArrayList<>();
        box.add(map);
        List<String> names = new ArrayList<>();
        names.add("pug");
        Image.drawBoundingBox(buf, box, names);
    }
}
