package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class DropoutGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("dropout");
        gh.fillNameDataAndVar(st, layer);

        gh.simpleFillTemplate(st, "prob", layer, "dropout_param.dropout_ratio", "0.5");

        return new GeneratorOutput(st.render(), 1);
    }
}
