package io.mxnet.caffetranslator;

import org.stringtemplate.v4.ST;

public class Optimizer {
    private final GenerationHelper gh;
    private final Solver solver;

    public Optimizer(Solver solver) {
        this.gh = new GenerationHelper();
        this.solver = solver;
    }

    public String generateInitCode() {
        ST st = gh.getTemplate("opt_" + solver.getType().toLowerCase());
        if (st == null) {
            System.err.println(String.format("Unknown optimizer type (%s). Using SGD instead.", solver.getType()));
            st = gh.getTemplate("opt_sgd");
        }

        st.add("solver", solver);
        return st.render();
    }
}
