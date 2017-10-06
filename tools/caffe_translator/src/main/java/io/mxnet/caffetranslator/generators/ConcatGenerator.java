package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class ConcatGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("concat");

        st.add("name", layer.getName());
        st.add("var", gh.getVarname(layer.getTop()));
        st.add("data", gh.getVarNames(layer.getBottoms()));

        String dim = layer.getAttr("concat_param.axis");
        if (dim != null) {
            st.add("dim", dim);
        }

        return new GeneratorOutput(st.render(), 1);
    }
}
