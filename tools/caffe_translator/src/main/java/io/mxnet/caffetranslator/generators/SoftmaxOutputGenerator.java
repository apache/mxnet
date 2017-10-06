package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class SoftmaxOutputGenerator extends BaseGenerator {
    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("softmaxoutput");
        gh.fillNameDataAndVar(st, layer);

        st.add("label", gh.getVarname(layer.getBottoms().get(1)));
        st.add("label_name", layer.getBottoms().get(1));

        return new GeneratorOutput(st.render(), 1);
    }
}
