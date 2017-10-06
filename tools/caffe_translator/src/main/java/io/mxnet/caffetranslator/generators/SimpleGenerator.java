package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import io.mxnet.caffetranslator.SymbolGenerator;
import lombok.Getter;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroup;
import org.stringtemplate.v4.STGroupFile;

import java.util.Iterator;
import java.util.Map;

public abstract class SimpleGenerator implements SymbolGenerator {

    @Getter
    STGroup stGroup;

    public SimpleGenerator() {
        stGroup = new STGroupFile("templates/symbols.stg");
    }

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {

        ST st = getStGroup().getInstanceOf(getTemplateName());
        Iterator it = getAttributeMap().entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry) it.next();
            String key = pair.getKey().toString();
            String value = pair.getValue().toString();

            String attr = layer.getAttr(value);
            st.add(key, attr);
        }

        fixupParams(st, layer, model);

        return new GeneratorOutput(st.render(), 1);
    }

    abstract Map<String, String> getAttributeMap();

    abstract String getTemplateName();

    protected void fixupParams(ST st, Layer layer, MLModel model) {
    }

}
