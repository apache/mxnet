package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class PluginLossGenerator extends BaseGenerator {

    private PluginLayerHelper helper;

    public PluginLossGenerator() {
        super();
        helper = new PluginLayerHelper();
    }

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("CaffePluginLossLayer");

        st.add("name", layer.getName());

        // Handle data
        if (layer.getBottoms().size() != 1)
            st.add("num_data", layer.getBottoms().size());
        String dataList = helper.getDataList(layer);
        st.add("data", dataList);

        // Set prototxt
        String prototxt = helper.makeOneLine(layer.getPrototxt());
        st.add("prototxt", prototxt);

        // Handle multiple outputs
        if (layer.getTops().size() > 1) {
            st.add("tops", layer.getTops());
            st.add("var", "out");
        } else if (layer.getTops().size() == 1) {
            st.add("var", layer.getTop());
        }

        return new GeneratorOutput(st.render(), 1);
    }

}
