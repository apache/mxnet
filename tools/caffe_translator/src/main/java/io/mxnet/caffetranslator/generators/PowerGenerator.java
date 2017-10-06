package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class PowerGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("power");

        String power = layer.getAttr("power_param.power", "1");
        String scale = layer.getAttr("power_param.scale", "1");
        String shift = layer.getAttr("power_param.shift", "0");

        st.add("var", gh.getVarname(layer.getTop()));
        st.add("data", gh.getVarname(layer.getBottom()));

        st.add("power", power);
        st.add("scale", scale);
        st.add("shift", shift);

        return new GeneratorOutput(st.render(), 1);
    }
}
