package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.List;

public class PermuteGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("permute");
        gh.fillNameDataAndVar(st, layer);

        List<String> axes = layer.getAttrList("permute_param.order");
        if (axes != null) {
            st.add("axes", axes);
        }

        return new GeneratorOutput(st.render(), 1);
    }
}
