package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenHelper;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.HashMap;
import java.util.Map;

public class AccuracyMetricsGenerator {

    Map<String, String> map;
    private GenHelper gh;

    public AccuracyMetricsGenerator() {
        map = new HashMap<>();
        gh = new GenHelper();
    }

    public String generate(MLModel model) {
        StringBuilder out = new StringBuilder();
        generateMap(model);

        for (Layer layer : model.getLayerList()) {
            if (layer.getType().equals("Accuracy")) {
                ST st;
                if (layer.getAttr("accuracy_param.top_k", "1").equals("1")) {
                    st = gh.getTemplate("accuracy");
                } else {
                    st = gh.getTemplate("top_k_accuracy");
                    st.add("k", layer.getAttr("accuracy_param.top_k"));
                }

                st.add("var", gh.getVarname(layer.getTop()));
                String outputName = map.get(layer.getBottoms().get(0)) + "_output";
                st.add("output_name", outputName);
                st.add("label_name", layer.getBottoms().get(1));
                st.add("name", layer.getName());

                out.append(st.render());
                out.append(System.lineSeparator());
            }
        }

        return out.toString();
    }

    private void generateMap(MLModel model) {
        for (Layer layer : model.getLayerList()) {
            // If this is not SoftmaxWithLoss, move on
            if (!layer.getType().equals("SoftmaxWithLoss")) {
                continue;
            }

            map.put(layer.getBottoms().get(0), layer.getName());
        }
    }
}
