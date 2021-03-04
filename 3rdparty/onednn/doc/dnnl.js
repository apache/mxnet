MathJax.Hub.Config({
    TeX: {
	Macros: {
	    src: '\\operatorname{src}',
	    srclayer: '\\operatorname{src\\_layer}',
	    srciter: '\\operatorname{src\\_iter}',
	    srciterc: '\\operatorname{src\\_iter\\_c}',
	    weights: '\\operatorname{weights}',
	    weightslayer: '\\operatorname{weights\\_layer}',
	    weightsiter: '\\operatorname{weights\\_iter}',
	    weightspeephole: '\\operatorname{weights\\_peephole}',
	    weightsprojection: '\\operatorname{weights\\_projection}',
	    bias: '\\operatorname{bias}',
	    dst: '\\operatorname{dst}',
	    dstlayer: '\\operatorname{dst\\_layer}',
	    dstiter: '\\operatorname{dst\\_iter}',
	    dstiterc: '\\operatorname{dst\\_iter\\_c}',
	    diffsrc: '\\operatorname{diff\\_src}',
	    diffsrclayer: '\\operatorname{diff\\_src\\_layer}',
	    diffsrciter: '\\operatorname{diff\\_src\\_iter}',
	    diffsrciterc: '\\operatorname{diff\\_src\\_iter\\_c}',
	    diffweights: '\\operatorname{diff\\_weights}',
	    diffweightslayer: '\\operatorname{diff\\_weights\\_layer}',
	    diffweightsiter: '\\operatorname{diff\\_weights\\_iter}',
	    diffweightspeephole: '\\operatorname{diff\\_weights\\_peephole}',
	    diffweightsprojection: '\\operatorname{diff\\_weights\\_projection}',
	    diffbias: '\\operatorname{diff\\_bias}',
	    diffdst: '\\operatorname{diff\\_dst}',
	    diffdstlayer: '\\operatorname{diff\\_dst\\_layer}',
	    diffdstiter: '\\operatorname{diff\\_dst\\_iter}',
	    diffdstiterc: '\\operatorname{diff\\_dst\\_iter\\_c}',
	    diffgamma: '\\operatorname{diff\\_\\gamma}',
	    diffbeta: '\\operatorname{diff\\_\\beta}',
	    workspace: '\\operatorname{workspace}'
	}
    }
});

MathJax.Ajax.loadComplete("[MathJax]/config/dnnl.js");
