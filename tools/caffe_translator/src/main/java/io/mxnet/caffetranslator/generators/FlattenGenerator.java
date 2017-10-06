package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class FlattenGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {

        ST st = getTemplate("flatten");
        gh.fillNameDataAndVar(st, layer);

        String axis = layer.getAttr("flatten_param.axis");
        if (axis != null && Integer.valueOf(axis) != 1) {
            String error = "Axis other that 1 is not supported for flatten" + System.lineSeparator();
            System.err.println(error);
            return new GeneratorOutput(error, 1);
        }

        return new GeneratorOutput(st.render(), 1);
    }
}
