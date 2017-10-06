package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenHelper;
import io.mxnet.caffetranslator.Layer;

public class PluginLayerHelper {

    GenHelper gh;

    public PluginLayerHelper() {
        gh = new GenHelper();
    }

    public String getDataList(Layer layer) {
        StringBuilder sb = new StringBuilder();
        int index = 0;

        if (layer.getBottoms().size() == 0) {
            return null;
        }

        for (String bottom : layer.getBottoms()) {
            sb.append("data_" + index + "=" + gh.getVarname(bottom) + ", ");
            index++;
        }
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 2);
        }
        return sb.toString();
    }

    public String makeOneLine(String prototxt) {
        prototxt = prototxt.replaceAll("\n", "").replaceAll("\r", "");
        prototxt = prototxt.replaceAll("'", "\'");
        prototxt = prototxt.replaceAll("\\s{2,}", " ").trim();
        return prototxt;
    }

}
