package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class BatchNormGenerator extends BaseGenerator {
    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("batchnorm");

        gh.fillNameDataAndVar(st, layer);

        if (layer.attrEquals("batch_norm_param.use_global_stats", "true")) {
            st.add("use_global_stats", true);
        }

        int layerIndex = layer.getLayerIndex();
        Layer nextLayer = model.getLayerList().get(layerIndex + 1);

        boolean nextLayerIsScale = false;
        if (nextLayer.getType().toLowerCase().equals("scale")) {
            String axis = nextLayer.getAttr("ScaleParameter.axis", "1");
            String numAxis = nextLayer.getAttr("ScaleParameter.num_axes", "1");
            if (axis.equals("1") && numAxis.equals("1")) {
                String biasTerm = nextLayer.getAttr("ScaleParameter.bias_term", "false");
                if (biasTerm.toLowerCase().equals("false")) {
                    nextLayerIsScale = true;
                }
            }
        }

        if (!nextLayerIsScale) {
            st.add("fix_beta", true);
            st.add("fix_gamma", true);
        }

        return new GeneratorOutput(st.render(), nextLayerIsScale ? 2 : 1);
    }
}
