package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenHelper;
import io.mxnet.caffetranslator.SymbolGenerator;
import org.stringtemplate.v4.ST;

import java.util.List;

public abstract class BaseGenerator implements SymbolGenerator {

    protected GenHelper gh;

    public BaseGenerator() {
        gh = new GenHelper();
    }

    protected ST getTemplate(String name) {
        return gh.getTemplate(name);
    }

    protected String generateVar(String varName, String symName, String lr_mult, String wd_mult, String init, List<Integer> shape) {
        ST st = getTemplate("var");
        st.add("var", varName);
        st.add("name", symName);

        st.add("lr_mult", (lr_mult == null) ? "None" : lr_mult);
        st.add("wd_mult", (wd_mult == null) ? "None" : wd_mult);
        st.add("init", (init == null) ? "None" : init);
        if (shape != null) st.add("shape", shape);

        return st.render();
    }

}
