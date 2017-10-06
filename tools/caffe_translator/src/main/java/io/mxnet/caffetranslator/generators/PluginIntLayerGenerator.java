package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class PluginIntLayerGenerator extends BaseGenerator {

    private PluginLayerHelper helper;


    public PluginIntLayerGenerator() {
        super();
        helper = new PluginLayerHelper();
    }

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        return generate(layer, model, 0);
    }

    public GeneratorOutput generate(Layer layer, MLModel model, int num_weight) {
        ST st = getTemplate("CaffePluginIntLayer");

        st.add("name", layer.getName());

        if (layer.getBottoms().size() != 1)
            st.add("num_data", layer.getBottoms().size());
        if (layer.getTops().size() != 1)
            st.add("num_out", layer.getTops().size());
        if (num_weight != 0)
            st.add("num_weight", num_weight);

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
            st.add("var", gh.getVarname(layer.getTop()));
        }

        return new GeneratorOutput(st.render(), 1);
    }

}
