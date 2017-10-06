package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.List;

public class EltwiseGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        String operation = layer.getAttr("eltwise_param.operation");
        if (operation == null) operation = "SUM";

        ST st;
        switch (operation) {
            case "SUM":
                st = getTemplate("add");
                break;
            case "PROD":
                st = getTemplate("mul");
                break;
            case "MAX":
                st = getTemplate("maximum");
                break;
            default:
                String error = "Unrecognized operation " + operation + " in Eltwise" + System.lineSeparator();
                System.err.print(error);
                return new GeneratorOutput(error, 1);
        }

        st.add("name", layer.getName());
        st.add("var", gh.getVarname(layer.getTop()));

        List<String> data = gh.getVarNames(layer.getBottoms());
        st.add("data1", data.get(0));
        st.add("data2", data.get(1));

        return new GeneratorOutput(st.render(), 1);
    }
}
