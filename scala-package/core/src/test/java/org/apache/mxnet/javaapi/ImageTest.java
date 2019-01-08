package org.apache.mxnet.javaapi;

import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import java.io.File;
import java.net.URL;

public class ImageTest {

    private String imLocation;

    private void downloadUrl(String url, String filePath, int maxRetry) throws Exception{
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
    public void downloadFile() throws Exception {
        String tempDirPath = System.getProperty("java.io.tmpdir");
        imLocation = tempDirPath + "/inputImages/Pug-Cookie.jpg";
        try {
            downloadUrl("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg",
                    imLocation, 3);
        } catch (Exception e) {
            throw e;
        }
    }

    @Test
    public void testImageProcess() {
        NDArray nd = Image.imRead(imLocation, 1, true, null);
        NDArray nd2 = Image.imResize(nd, 224, 224, null, null);
        NDArray cropped = Image.fixedCrop(nd, 0, 0, 224, 224);
        Image.toImage(cropped);
    }
}
