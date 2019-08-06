# R script to convert .Rd documentation to Sphinx Restructured Text .rst files:

source("./Rd2RstFunctions.R") # to load helper functions.

### Runtime parameters to specify: ###
rd_dir = './man/' # directory of .Rd documentation files from man/ subdirectory in compiled R package.
rstfiles_dir = './doc' # Directory where to store index.rst and other Sphinx toctrees for R documentation.
rst_subdir = '' # Subdirectory of the above directory where to store R functions.
srcsite = "http://github.com/apache/incubator-mxnet/blob/master/" # Where source code is located for links.
ignore_names = c("mxnet","mxnet.export") # File-names to ignore when generating rst files.

printorder = c('title','description','usage','arguments','value','srclink') # details is merged with description.
name_tag = "\\name{"
title_tag = "\\title{"
usage_tag = "\\usage{"
arguments_tag = "\\arguments{"
value_tag = "\\value{"
description_tag = "\\description{"
details_tag = "\\details{"
examples_tag = "\\examples{"
format_tag = "\\format{"
alias_tag = "\\alias{"
keyword_tag= "\\keyword{"
doctype_tag = "\\docType{"
unused_tagpatterns = c(format_tag,alias_tag,keyword_tag,doctype_tag) # Extra tags that we need to know about for parsing, but are not used in RST output.
rstheader = '.. raw:: html

' # header for each RST function file.
rstfooter = "\n\n.. disqus::" # footer.

### End of runtime parameters ###


rst_dir = paste(rstfiles_dir,rst_subdir,sep='')
indexrst_file = paste0(rstfiles_dir,'index.rst')
filenames = list.files(rd_dir, pattern = "*.Rd")
func_names = sapply(filenames, FUN=function(x) substr(x,1,nchar(x)-3), USE.NAMES = FALSE)
func_names = func_names[!(func_names %in% ignore_names)]
func_desc = rep("",length(func_names))  # descriptions of each function.


# Clear out RST files subdirectory:
file.remove(dir(rst_dir, pattern = "*", full.names = TRUE))

# Get markdown files and then convert to Restructured Text format:
for (i in 1:length(func_names)) {
    func_name = func_names[i]
    print(sprintf("processing function #%d out of %d: %s",i, length(func_names),func_name))
    rdfile = paste(rd_dir,func_name,'.Rd',sep='')
    rdcontents = readLines(rdfile)
    rdcontents = formatInlineTags(rdcontents)
    results = list()
    for (sec in printorder) {
        results[['sec']] = ""
    }
    name_line = grep(name_tag, rdcontents, fixed= TRUE)
    title_line = grep(title_tag, rdcontents, fixed = TRUE)
    usage_line = grep(usage_tag, rdcontents, fixed = TRUE)
    arguments_line = grep(arguments_tag, rdcontents, fixed = TRUE)
    value_line = grep(value_tag, rdcontents, fixed = TRUE)
    description_line = grep(description_tag, rdcontents, fixed = TRUE)
    details_line = grep(details_tag, rdcontents, fixed=TRUE)
    examples_line = grep(examples_tag, rdcontents, fixed=TRUE)
    tag_lines = c(name_line, title_line, usage_line, arguments_line, value_line, 
                  description_line, details_line, examples_line)
    for (tag_pattern in unused_tagpatterns) {
        pattern_line = grep(tag_pattern, rdcontents, fixed = TRUE)
        tag_lines = c(tag_lines, pattern_line)
    }
    if (length(tag_lines) == 0) {
        stop("Rd file is wrongly formatted. No known tags found.")
    }
    if (length(name_line) == 1) { # check that file name matches name used in "\name{}" tag.
        name_string = trimws(getTagText(name_tag, name_line, tag_lines, rdcontents))
        if (name_string != func_name) { # \name{} tag overrides file name.
            func_name = name_string
            func_names[i] = name_string
        }
    }
    rstfile = paste(rst_dir,func_name,'.rst',sep='')
    # Title:
    results$title = underbarTitle(paste0("``",func_name,"``"),symbol="=",extra_newline=F)
    
    # Description:
    if (length(title_line) == 1) {
        title_string = getTagText(title_tag, title_line, tag_lines, rdcontents)
        subfuncname = strsplit(func_name, "\\.")[[1]][length(strsplit(func_name, "\\.")[[1]])]
        short_desc = trimws(gsub(".*`.*`:","", title_string)) # ignore function name and only keep description.
        short_desc = trimws(gsub(paste(func_name,":",sep=''), "", short_desc, fixed=T))
        short_desc = trimws(gsub(paste(subfuncname,":",sep=''), "", short_desc, fixed=T))
        short_desc = paste(short_desc,collapse="\n")
    } else {
        short_desc = ""
    }
    func_desc[i] = gsub("\\.$","",trimws(short_desc))
    # Rest of description field:
    if (length(description_line) == 1) {
        description_string = getTagText(description_tag, description_line, tag_lines, rdcontents)
        if (!grepl(tolower(short_desc), tolower(paste(description_string,collapse="\n")), fixed=TRUE)) {
                # Short Description from \title{} is not redundant, so should be added to official description.
            formatted_description = c(punctuate(short_desc), "", description_string)
        } else {
            formatted_description = description_string
        }
    } else {
        formatted_description = short_desc
    }
    # Merge Details into description: 
    if (length(details_line) == 1) {
        details_string = getTagText(details_tag, details_line, tag_lines, rdcontents)
        formatted_description = c(formatted_description, "", details_string)
    }
    # Merge Examples into description:
    if (length(examples_line) == 1) {
        examples_string = getTagText(examples_tag, examples_line, tag_lines, rdcontents)
        examples_string = paste0("\t",trimws(examples_string))
        formatted_description = c(formatted_description, "","**Example**::","",examples_string)
    }
    results$description = c(underbarTitle("Description"),formatted_description)
    # Usage:
    if (length(usage_line) == 1) {
        usage_string = getTagText(usage_tag, usage_line, tag_lines, rdcontents)
        results$usage = c(underbarTitle("Usage"),".. code:: r", paste0("\n\t",usage_string))
    }
    
    # Value:
    if (length(value_line) == 1) {
        value_string = getTagText(value_tag, value_line, tag_lines, rdcontents)
        words = strsplit(value_string,split=" ")[[1]]
        if (length(words) > 1) {
            if (grepl("[[:upper:]]",substr(words[2],1,1))) {
                val_name = paste0("``", words[1], "``")
                value_string = paste(c(val_name, words[2:length(words)]), collapse=" ")
            }
        }
        results$value = c(underbarTitle("Value"),value_string)
    }
    
    # Arguments:
    if (length(arguments_line) == 1) {
        args_string = getTagText(arguments_tag, arguments_line, tag_lines, rdcontents)
        item_pattern = "\\item{"
        argname_pattern = "\\\\item\\{.*?\\}"
        item_locs = grep(item_pattern, args_string, fixed=T)
        if (length(item_locs) > 0) {
            LHS_WIDTH = 40
            RHS_WIDTH = 60
            lhs_dashes = paste0(rep('-',LHS_WIDTH),collapse='')
            rhs_dashes = paste0(rep('-',RHS_WIDTH),collapse='')
            hbar = paste0("+",lhs_dashes,"+",rhs_dashes,"+")
            titlebar = paste0("+",paste0(rep('=',LHS_WIDTH),collapse=''),"+",
                              paste0(rep('=',RHS_WIDTH),collapse=''),"+")
            args_formatted = c(hbar, argsTableRow("Argument", "Description",LHS_WIDTH,RHS_WIDTH), titlebar)
            for (item_loc in item_locs) {
                item_line = args_string[item_loc]
                regmatch = regmatches(item_line,regexpr(argname_pattern,item_line))
                if (length(regmatch) != 1) {
                    stop(paste("Badly formatted \\item in Rd file:", item_line))
                }
                arg_name = trimws(substr(trimws(regmatch),nchar(item_pattern)+1,nchar(regmatch)-1))
                arg_name = trimws(gsub("[[:punct:]]+$", "", arg_name)) # remove punctuation at end of argument name. NOTE: no argument-name is allowed to end in punctuation!
                item_endloc = min(item_locs[which(item_locs > item_loc)]-1, length(args_string))
                arg_desc = trimws(gsub(paste0(argname_pattern,"\\{"),"",args_string[item_loc:item_endloc]))
                while (nchar(trimws(arg_desc[length(arg_desc)])) == 0) {
                    arg_desc = arg_desc[-length(arg_desc)]
                }
                lastline = trimws(arg_desc[length(arg_desc)])
                if (substr(lastline,nchar(lastline),nchar(lastline)) == '}') {
                    arg_desc[length(arg_desc)] = substr(lastline,1,nchar(lastline)-1)
                }
                # Check whether first line of arg_desc is about type of argument.
                firstline = trimws(arg_desc[1])
                if (nchar(trimws(firstline)) > 0) {
                    secondLineCap = FALSE
                    if (length(arg_desc) > 1) {
                        secondline = trimws(arg_desc[2])
                        secondLineCap = grepl('[[:upper:]]',substr(secondline,1,1))
                    }
                    argEvidence = (grepl("default=.*?[0-9][[:punct:]]*$",firstline) || 
                                   grepl("default=[[:punct:]]*$",firstline) || 
                                   grepl("optional[[:punct:]]*$",firstline) ||
                                   grepl("required[[:punct:]]*$",firstline) ||
                                   grepl("float[[:punct:]]*$",firstline) ||
                                   grepl("string[[:punct:]]*$",firstline)
                                  )
                    if ((length(arg_desc) > 1) && (secondLineCap || argEvidence)) {
                        firstline = punctuate(firstline)
                        arg_desc = c(firstline,"",arg_desc[2:length(arg_desc)])
                    }
                }
                # arg_desc = paste(arg_desc,collapse="\n")
                new_item = argsTableRow(paste0("``",arg_name,"``"), arg_desc, LHS_WIDTH, RHS_WIDTH)
                args_formatted = c(args_formatted,new_item,hbar)
            }
        } else {
            args_formatted = args_string
        }
        results$arguments = c(underbarTitle("Arguments"), args_formatted)
    }
    rstcontents = rstheader
    for (sec in printorder) {
        rstcontents = c(rstcontents, "", results[[sec]])
    }
    # Postprocess rstcontents:
    # - TODO?: insert links in func_names
    
    # Format math:
    math_pattern = ".. math::"
    math_indices = grep(math_pattern, rstcontents, fixed=T)
    lines_added = 0
    for (i in math_indices) {
        i = i + lines_added
        math_line = rstcontents[i]
        if (rstcontents[i+1] != "") { # add blank line after .. math::
                rstcontents = c(rstcontents[1:i],"",rstcontents[(i+1):length(rstcontents)])
                lines_added = lines_added + 1 
        }
        if (!grepl("^[[:space:]]",rstcontents[i+2])) { # add tab to first mathematical content line
            rstcontents[i+2] = paste0("\t",rstcontents[i+2])
        }
    }
    rstcontents = formatRSTindentation(rstcontents) # indents blocks preceded by ::
    rstcontents = fixBulletDoc(rstcontents) # TODO: this is a problem in underlying documentation. Should not be need once underyling problem is fixed.
    
    # Create source code links for "Defined in" at the end of RST doc:
    pattern = "^Defined in"
    defindex = grep(pattern, rstcontents)
    if (length(defindex) == 1) {
        defline = rstcontents[defindex]
        defloc = gregexpr("Defined in", defline)[[1]][1]
        linkstr = trimws(substr(defline,defloc+nchar(pattern),nchar(defline)))
        newdefline = paste0("Link to Source Code: ", createlink(linkstr, srcsite))
        rstcontents = c(rstcontents[-defindex], "", newdefline)
    }
    disqus_id = paste0("   :disqus_identifier: ",func_name)
    rstcontents = c(rstcontents, rstfooter,disqus_id)
    writeLines(rstcontents, rstfile)
}


### RST toctree files ###

# Categorize functions to form toctree groupings:

manipulation_patterns = "apply|as\\.|\\.where|ops\\.|cast|copy|fill|choose|concat|crop|\\.diag|expand|flatten|flip|gather|grid|argmax|argmin|pad|pick|shuffle|reshape|repeat|reverse|load|save|scatter|sequence|slice|sort|split|swap|squeeze|stack|space.to.depth|depth.to.space|take|tile|transpose|upsampling|unravel|index|broadcast|ravel|mask|topk"
neuralnet_patterns = "activation|batchnorm|bilinearsampler|blockgrad|convolution|deconvolution|correlation|ctc|custom|dropout|embedding|fullyconnected|lrn|klsparsereg|instancenorm|layernorm|relu|regression|loss|pool|rnn|soft|sigmoid|transformer|gradient|svm"

ndarray_inds = grep("^mx\\.nd\\.|ndarray",tolower(func_names))
ndarray_inds = ndarray_inds[!(ndarray_inds %in% grep("update|random|sample|mx.nd.normal|rmsprop|\\.uniform",tolower(func_names)))]
ndarray_funcs = func_names[ndarray_inds]
nd_attribute_inds = grep("dim\\.|length|\\.shape|\\.size|print\\.|is\\.",tolower(ndarray_funcs))
nd_create_inds = grep("zeros|ones|mx\\.nd\\.array|one.hot",tolower(ndarray_funcs))
nd_manipulation_inds = grep(manipulation_patterns,tolower(ndarray_funcs))
nd_neuralnet_inds = grep(neuralnet_patterns,tolower(ndarray_funcs))
nd_attribute_funcs = ndarray_funcs[nd_attribute_inds]
nd_create_funcs = ndarray_funcs[nd_create_inds]
nd_manipulation_funcs = ndarray_funcs[nd_manipulation_inds]
nd_neuralnet_funcs = ndarray_funcs[nd_neuralnet_inds]
nd_math_funcs = ndarray_funcs[-unique(c(nd_attribute_inds,nd_create_inds,nd_manipulation_inds,nd_neuralnet_inds))]

nd_manipulation_conversion_funcs = ndarray_funcs[grep("\\.cast|copy|as\\.|tostype|load|save|ops\\.",tolower(ndarray_funcs))]
nd_manipulation_shape_funcs = ndarray_funcs[grep("flatten|reshape|scatter|split|stack|squeeze|concat",tolower(ndarray_funcs))]
nd_manipulation_expand_funcs = ndarray_funcs[grep("broadcast|pad|repeat|tile|upsamp|expand|grid",tolower(ndarray_funcs))]
nd_manipulation_rearrange_funcs = ndarray_funcs[grep("reverse|shuffle|transpose|flip|space.to.depth|depth.to.space|swapax",tolower(ndarray_funcs))]
nd_manipulation_sort_funcs = ndarray_funcs[grep("sort|search|where|find|topk|argmax|argmin",tolower(ndarray_funcs))]
nd_manipulation_index_funcs = ndarray_funcs[grep("pick|take|slice|index|\\.diag|crop|last|gather|mask",tolower(ndarray_funcs))]
nd_manipulation_subfuncs = c(nd_manipulation_conversion_funcs,nd_manipulation_shape_funcs,
                             nd_manipulation_expand_funcs,nd_manipulation_rearrange_funcs,nd_manipulation_sort_funcs,
                             nd_manipulation_index_funcs)
nd_manipulation_rest_funcs = nd_manipulation_funcs[!(nd_manipulation_funcs %in% nd_manipulation_subfuncs)]
if (length(nd_manipulation_rest_funcs) > 0) {
    warning(paste("The following nd_manipulation_funcs will not be included in documentation: \n",
                  paste(nd_manipulation_rest_funcs,collapse=" ")))
}
nd_math_round_funcs = ndarray_funcs[grep("trunc|round|ceil|floor|fix|clip|\\.rint",tolower(ndarray_funcs))]
nd_math_reduction_funcs = ndarray_funcs[grep("mean|nd\\.sum|nansum|\\.prod|nanprod|nd\\.min|nd\\.max",tolower(ndarray_funcs))]
nd_math_linalg_funcs = ndarray_funcs[grep("\\.dot|batch_dot|linalg|\\.norm|l2|l1|frobenius|khatri.rao",tolower(ndarray_funcs))]
nd_math_trig_funcs = ndarray_funcs[grep("arccos$|arcsin$|arctan$|sin$|cos$|tan$|radians|\\.degrees",tolower(ndarray_funcs))]
nd_math_hyperbolic_funcs = ndarray_funcs[grep("cosh|sinh|tanh",tolower(ndarray_funcs))]
nd_math_subfuncs = c(nd_math_round_funcs, nd_math_reduction_funcs, nd_math_linalg_funcs,
                     nd_math_trig_funcs, nd_math_hyperbolic_funcs)
nd_math_arithmetic_funcs = nd_math_funcs[!(nd_math_funcs %in% nd_math_subfuncs)]


symbol_inds = grep("symbol|arguments|mx.apply|outputs|internals|children|graph.viz", tolower(func_names))
symbol_inds = symbol_inds[!(symbol_inds %in% grep("update|random|sample|\\.normal|\\.uniform|rmsprop|mx.simple.bind",tolower(func_names)))]
symbol_funcs = func_names[symbol_inds]
symbol_attribute_inds = grep("graph|outputs|internals|children|arguments|dim\\.|length|\\.shape|\\.size|print\\.|is\\.",tolower(symbol_funcs))
symbol_create_inds = grep("group|zeros|ones|as\\.symbol|mx\\.nd\\.symbol|mx.symbol.variable|one.hot",tolower(symbol_funcs))
symbol_manipulation_inds = grep(manipulation_patterns, tolower(symbol_funcs))
symbol_neuralnet_inds = grep(neuralnet_patterns,tolower(symbol_funcs))
symbol_attribute_funcs = symbol_funcs[symbol_attribute_inds]
symbol_create_funcs = symbol_funcs[symbol_create_inds]
symbol_manipulation_funcs = symbol_funcs[symbol_manipulation_inds]
symbol_neuralnet_funcs = symbol_funcs[symbol_neuralnet_inds]
symbol_math_funcs = symbol_funcs[-unique(c(symbol_attribute_inds,symbol_create_inds,symbol_manipulation_inds,symbol_neuralnet_inds))]

symbol_manipulation_conversion_funcs = symbol_funcs[grep("apply|\\.cast|copy|as\\.|tostype|load|save|ops\\.",tolower(symbol_funcs))]
symbol_manipulation_shape_funcs = symbol_funcs[grep("flatten|reshape|scatter|split|stack|squeeze|concat",tolower(symbol_funcs))]
symbol_manipulation_expand_funcs = symbol_funcs[grep("broadcast|pad|repeat|tile|upsamp|expand|grid",tolower(symbol_funcs))]
symbol_manipulation_rearrange_funcs = symbol_funcs[grep("reverse|shuffle|transpose|flip|space.to.depth|depth.to.space|swapax",tolower(symbol_funcs))]
symbol_manipulation_sort_funcs = symbol_funcs[grep("sort|search|where|find|topk|argmax|argmin",tolower(symbol_funcs))]
symbol_manipulation_index_funcs = symbol_funcs[grep("pick|take|slice|index|\\.diag|crop|last|gather|mask",tolower(symbol_funcs))]
symbol_manipulation_subfuncs = c(symbol_manipulation_conversion_funcs,symbol_manipulation_shape_funcs,
                             symbol_manipulation_expand_funcs,symbol_manipulation_rearrange_funcs,symbol_manipulation_sort_funcs,
                             symbol_manipulation_index_funcs)
symbol_manipulation_rest_funcs = symbol_manipulation_funcs[!(symbol_manipulation_funcs %in% symbol_manipulation_subfuncs)]
if (length(symbol_manipulation_rest_funcs) > 0) {
    warning(paste("The following symbol_manipulation_funcs will not be included in documentation: \n",
                  paste(symbol_manipulation_rest_funcs,collapse=" ")))
}
symbol_math_round_funcs = symbol_funcs[grep("trunc|round|ceil|floor|fix|clip|\\.rint",tolower(symbol_funcs))]
symbol_math_reduction_funcs = symbol_funcs[grep("mean|symbol\\.sum|nansum|\\.prod|nanprod|symbol\\.min|symbol\\.max",tolower(symbol_funcs))]
symbol_math_linalg_funcs = symbol_funcs[grep("\\.dot|batch_dot|linalg|\\.norm|l1|l2|frobenius|khatri.rao",tolower(symbol_funcs))]
symbol_math_trig_funcs = symbol_funcs[grep("arccos$|arcsin$|arctan$|sin$|cos$|tan$|radians|\\.degrees",tolower(symbol_funcs))]
symbol_math_hyperbolic_funcs = symbol_funcs[grep("cosh|sinh|tanh",tolower(symbol_funcs))]
symbol_math_subfuncs = c(symbol_math_round_funcs, symbol_math_reduction_funcs, symbol_math_linalg_funcs,
                         symbol_math_trig_funcs, symbol_math_hyperbolic_funcs)
symbol_math_arithmetic_funcs = symbol_math_funcs[!(symbol_math_funcs %in% symbol_math_subfuncs)]


io_inds = unique(c(grep(".io",tolower(func_names),fixed=TRUE), grep("dataiter", tolower(func_names)), 
                 grep("im2rec",tolower(func_names))))
io_funcs = func_names[io_inds]
imageio_funcs = io_funcs[unique(c(grep("image", tolower(io_funcs)),grep("im2rec", tolower(io_funcs))))]
io_funcs = io_funcs[-unique(c(grep("image", tolower(io_funcs)),grep("im2rec", tolower(io_funcs))))]

context_inds = unique(c(grep("cpu",tolower(func_names)), grep("gpu",tolower(func_names)),
                      grep("ctx",tolower(func_names)),grep("context",tolower(func_names))
                     ))
context_funcs = func_names[context_inds]
model_inds = unique(c(grep("model",tolower(func_names)), grep("serialize",tolower(func_names)),
               grep("mlp",tolower(func_names)), grep("rnn",tolower(func_names)),
               grep("mx.infer",tolower(func_names))))
model_funcs = func_names[model_inds]
metric_inds = grep("metric",tolower(func_names))
metric_funcs = func_names[metric_inds]

optimization_inds = grep("mx\\.opt|mx\\.lr|adam|init|ftml|ftrl|sgd|signum|\\.update",tolower(func_names))
optimization_inds = optimization_inds[!(optimization_inds %in% grep("\\.exec",tolower(func_names)))]

optimization_funcs = func_names[optimization_inds]
optimization_opt_funcs = optimization_funcs[grep(".opt",optimization_funcs,fixed=T)]
optimization_init_funcs = optimization_funcs[grep(".init",optimization_funcs,fixed=T)]
optimization_lr_funcs = optimization_funcs[grep(".lr",optimization_funcs,fixed=T)]
optimization_rest_funcs = optimization_funcs[which(!(optimization_funcs %in% c(optimization_opt_funcs,optimization_init_funcs,optimization_lr_funcs)))]
optimization_symupdate_funcs = optimization_rest_funcs[grep("symbol",tolower(optimization_rest_funcs))]
optimization_ndupdate_funcs = optimization_rest_funcs[!(optimization_rest_funcs %in% optimization_symupdate_funcs)]

random_inds = grep("sample|mx.nd.normal|random|runif|rnorm|seed|uniform$",tolower(func_names))
random_funcs = func_names[random_inds]
random_funcs = random_funcs[!grepl("layernorm|mx.init|bilinear",tolower(random_funcs))]
random_sym_funcs = random_funcs[unique(grep(".symbol",random_funcs,fixed=T))]
random_nd_funcs = random_funcs[unique(grep(".nd",random_funcs,fixed=T))]
random_rest_funcs = random_funcs[which(!(random_funcs %in% c(random_sym_funcs,random_nd_funcs)))]

exec_inds = grep("exec|mx.simple.bind",tolower(func_names))
exec_funcs = func_names[exec_inds]
kv_inds = grep("kv",tolower(func_names))
kv_funcs = func_names[kv_inds]
advanced_inds = unique(c(exec_inds,kv_inds))
advanced_funcs = func_names[advanced_inds]

profiler_inds = grep("profiler",tolower(func_names))
profiler_funcs = func_names[profiler_inds]
callback_inds = grep("callback",tolower(func_names))
callback_funcs = func_names[callback_inds]


# Write toctree files:
disqusfooter = ""

index_rstheader = 'MXNet API for R 
======================

This section contains documentation for the R version of MXNet.

To instead view documentation directly in your R console, enter:

.. code:: r

    library(mxnet); help(mxnet)

and then click on ``Index`` in the Help window that appears.




.. toctree::
   :maxdepth: 2
'
index_rstnames = paste0(rst_subdir, c('ndarray','symbol', 'io', 'context', 
                            'model', 'metric', 'optimization', 'random',
                            'monitoring', 'advanced', 'allfunctions'))
index_rstcontents = paste0("   ",index_rstnames)

index_rstdescriptions = c("NDArray",
        "Symbol",
        "IO",
        "Context",
        "Model",
        "Metric", 
        "Optimization", 
        "Random Sampling",
        "Monitoring",
        "Advanced",
        "Complete Library"
    )
index_rstformatted = paste0("   ", index_rstdescriptions, " <",index_rstnames,">")
disqus_id = paste0("   :disqus_identifier: ",'api')
writeLines(c(index_rstheader, index_rstformatted, rstfooter,disqus_id), paste0(rstfiles_dir,"index.rst"))

# Create ndarray.rst:
ndarray_rstheader = "NDArray 
====================================================

The core MXNet data structure for all mathematical operations
------------------------------------------------------------------

"
ndarray_attribute_header = "
Get array attributes
--------------------

"
ndarray_create_header = "
Create arrays
------------------

"
ndarray_manipulation_header = "
Manipulation of arrays
------------------------

"
nd_manipulation_conversion_header = "
Conversion
^^^^^^^^^^^^

"
nd_manipulation_shape_header = "
Reshaping
^^^^^^^^^^^^^

"
nd_manipulation_expand_header = "
Expanding elements
^^^^^^^^^^^^^^^^^^^^^

" 
nd_manipulation_rearrange_header = "
Rearranging elements
^^^^^^^^^^^^^^^^^^^^^^

" 
nd_manipulation_sort_header = "
Sorting and searching
^^^^^^^^^^^^^^^^^^^^^^^

"
nd_manipulation_index_header = "
Indexing
^^^^^^^^^^^^

"
ndarray_neuralnet_header ="
Neural network array operations
-----------------------------------------------------

"
ndarray_math_header = "
Mathematical operations on arrays 
-----------------------------------

"
nd_math_arithmetic_header = "
Arithmetic
^^^^^^^^^^^^^

"
nd_math_reduction_header = "
Reduce
^^^^^^^^^^^^^

"
nd_math_round_header = "
Round
^^^^^^^^^^^^^

"
nd_math_linalg_header = "
Linear algebra
^^^^^^^^^^^^^^^^

"
nd_math_trig_header = "
Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^

"
nd_math_hyperbolic_header = "
Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^

"
disqus_id = paste0("   :disqus_identifier: ",'ndarray')
writeLines(c(ndarray_rstheader, ndarray_attribute_header, formatFuncsToctree(nd_attribute_funcs,func_names,func_desc),
    ndarray_create_header, formatFuncsToctree(nd_create_funcs,func_names,func_desc),
    ndarray_manipulation_header, nd_manipulation_conversion_header, formatFuncsToctree(nd_manipulation_conversion_funcs,func_names,func_desc),
    nd_manipulation_shape_header, formatFuncsToctree(nd_manipulation_shape_funcs,func_names,func_desc),
    nd_manipulation_expand_header, formatFuncsToctree(nd_manipulation_expand_funcs,func_names,func_desc),
    nd_manipulation_rearrange_header, formatFuncsToctree(nd_manipulation_rearrange_funcs,func_names,func_desc),
    nd_manipulation_sort_header, formatFuncsToctree(nd_manipulation_sort_funcs,func_names,func_desc),
    nd_manipulation_index_header, formatFuncsToctree(nd_manipulation_index_funcs,func_names,func_desc),
    ndarray_math_header, nd_math_arithmetic_header, formatFuncsToctree(nd_math_arithmetic_funcs,func_names,func_desc),
    nd_math_reduction_header, formatFuncsToctree(nd_math_reduction_funcs,func_names,func_desc),
    nd_math_round_header, formatFuncsToctree(nd_math_round_funcs,func_names,func_desc),
    nd_math_linalg_header, formatFuncsToctree(nd_math_linalg_funcs,func_names,func_desc),
    nd_math_trig_header, formatFuncsToctree(nd_math_trig_funcs,func_names,func_desc),
    nd_math_hyperbolic_header, formatFuncsToctree(nd_math_hyperbolic_funcs,func_names,func_desc),
    ndarray_neuralnet_header, formatFuncsToctree(nd_neuralnet_funcs,func_names,func_desc),
    rstfooter, disqus_id), paste0(rst_dir,"ndarray.rst"))


symbol_rstheader = "Symbol
====================================================

Symbolic programming with computation graphs 
------------------------------------------------------

"
symbol_attribute_header = "
Get symbol attributes
--------------------

"
symbol_create_header = "
Create symbols
------------------

"
symbol_manipulation_header = "
Manipulation of symbols
------------------------

"
symbol_manipulation_conversion_header = "
Conversion
^^^^^^^^^^^^

"
symbol_manipulation_shape_header = "
Reshaping
^^^^^^^^^^^^^

"
symbol_manipulation_expand_header = "
Expanding elements
^^^^^^^^^^^^^^^^^^^^^

" 
symbol_manipulation_rearrange_header = "
Rearranging elements
^^^^^^^^^^^^^^^^^^^^^^

" 
symbol_manipulation_sort_header = "
Sorting and searching
^^^^^^^^^^^^^^^^^^^^^^^

"
symbol_manipulation_index_header = "
Indexing
^^^^^^^^^^^^

"
symbol_neuralnet_header ="
Neural network symbol operations
-----------------------------------------------------

"
symbol_math_header = "
Mathematical operations on symbols 
-----------------------------------

"
symbol_math_arithmetic_header = "
Arithmetic
^^^^^^^^^^^^^

"
symbol_math_reduction_header = "
Reduce
^^^^^^^^^^^^^

"
symbol_math_round_header = "
Round
^^^^^^^^^^^^^

"
symbol_math_linalg_header = "
Linear algebra
^^^^^^^^^^^^^^^^

"
symbol_math_trig_header = "
Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^

"
symbol_math_hyperbolic_header = "
Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^

"
disqus_id = paste0("   :disqus_identifier: ",'symbol')
writeLines(c(symbol_rstheader, symbol_attribute_header, formatFuncsToctree(symbol_attribute_funcs,func_names,func_desc),
    symbol_create_header, formatFuncsToctree(symbol_create_funcs,func_names,func_desc),
    symbol_manipulation_header, symbol_manipulation_conversion_header, formatFuncsToctree(symbol_manipulation_conversion_funcs,func_names,func_desc),
    symbol_manipulation_shape_header, formatFuncsToctree(symbol_manipulation_shape_funcs,func_names,func_desc),
    symbol_manipulation_expand_header, formatFuncsToctree(symbol_manipulation_expand_funcs,func_names,func_desc),
    symbol_manipulation_rearrange_header, formatFuncsToctree(symbol_manipulation_rearrange_funcs,func_names,func_desc),
    symbol_manipulation_sort_header, formatFuncsToctree(symbol_manipulation_sort_funcs,func_names,func_desc),
    symbol_manipulation_index_header, formatFuncsToctree(symbol_manipulation_index_funcs,func_names,func_desc),
    symbol_math_header, symbol_math_arithmetic_header, formatFuncsToctree(symbol_math_arithmetic_funcs,func_names,func_desc),
    symbol_math_reduction_header, formatFuncsToctree(symbol_math_reduction_funcs,func_names,func_desc),
    symbol_math_round_header, formatFuncsToctree(symbol_math_round_funcs,func_names,func_desc),
    symbol_math_linalg_header, formatFuncsToctree(symbol_math_linalg_funcs,func_names,func_desc),
    symbol_math_trig_header, formatFuncsToctree(symbol_math_trig_funcs,func_names,func_desc),
    symbol_math_hyperbolic_header, formatFuncsToctree(symbol_math_hyperbolic_funcs,func_names,func_desc),
    symbol_neuralnet_header, formatFuncsToctree(symbol_neuralnet_funcs,func_names,func_desc),
            rstfooter,disqus_id), paste0(rst_dir,"symbol.rst"))



io_rstheader = "IO
====================================================

Efficient distributed data loading and augmentation 
------------------------------------------------------

"
io_rstheader2 = "
Loading image data 
---------------------

"
disqus_id = paste0("   :disqus_identifier: ",'io')
writeLines(c(io_rstheader, formatFuncsToctree(io_funcs,func_names,func_desc),
             io_rstheader2, formatFuncsToctree(imageio_funcs,func_names,func_desc),
            rstfooter,disqus_id), paste0(rst_dir,"io.rst"))


context_rstheader = "Context
====================================================

Manage device type and id where computations are carried out
-----------------------------------------------------------------

"
disqus_id = paste0("   :disqus_identifier: ",'context')
writeLines(c(context_rstheader, formatFuncsToctree(context_funcs,func_names,func_desc),
             rstfooter, disqus_id), paste0(rst_dir,"context.rst"))

model_rstheader = "Model
====================================================

Create, load/save, and use MXNet models
------------------------------------------------

"
disqus_id = paste0("   :disqus_identifier: ",'model')
writeLines(c(model_rstheader, formatFuncsToctree(model_funcs,func_names,func_desc),
             rstfooter, disqus_id), paste0(rst_dir,"model.rst"))

metric_rstheader = "Metric
====================================================

Evaluation metrics to measure model performance
--------------------------------------------------
"
disqus_id = paste0("   :disqus_identifier: ",'io')
writeLines(c(metric_rstheader, formatFuncsToctree(metric_funcs,func_names,func_desc),
             rstfooter, disqus_id), paste0(rst_dir,"metric.rst"))


optimization_rstheader = "Optimization
====================================================

Initialize and update model weights during training
--------------------------------------------------------------

"
optimization_rst_opt = "
Optimizers
---------------------------

"
optimization_rst_init = "
Initialization
---------------------------

"
optimization_rst_lr = "
Learning rate schedule
---------------------------

"
optimization_rst_ndupdate = "
Optimizer updates (NDArray)
------------------------------------------

"
optimization_rst_symupdate = "
Optimizer updates (Symbol)
------------------------------------------

"
disqus_id = paste0("   :disqus_identifier: ",'optimization')
writeLines(c(optimization_rstheader,optimization_rst_opt, formatFuncsToctree(optimization_opt_funcs,func_names,func_desc),
             optimization_rst_init, formatFuncsToctree(optimization_init_funcs,func_names,func_desc),
             optimization_rst_lr, formatFuncsToctree(optimization_lr_funcs,func_names,func_desc),
             optimization_rst_ndupdate, formatFuncsToctree(optimization_ndupdate_funcs,func_names,func_desc),
             optimization_rst_symupdate, formatFuncsToctree(optimization_symupdate_funcs,func_names,func_desc),
            rstfooter, disqus_id), paste0(rst_dir,"optimization.rst"))


random_rstheader = "Random Sampling
====================================================

"
random_rst_rest = "
Random number generation in MXNet 
-----------------------------------

"
random_rst_nd = "
Random NDArrays
-----------------

"
random_rst_sym = "
Random Symbols
------------------

"
disqus_id = paste0("   :disqus_identifier: ",'random')
writeLines(c(random_rstheader, random_rst_rest, formatFuncsToctree(random_rest_funcs,func_names,func_desc),
             random_rst_nd, formatFuncsToctree(random_nd_funcs,func_names,func_desc),
             random_rst_sym, formatFuncsToctree(random_sym_funcs,func_names,func_desc), 
            rstfooter, disqus_id), paste0(rst_dir,"random.rst"))

advanced_rstheader = "Advanced
====================================================

CAUTION: This section is only intended for advanced users. 
Direct interactions with KVStore and Executor are dangerous and not recommended.


Key-Value Store: Operate over multiple devices (GPUs) on a single device
--------------------------------------------------------------------------------------------------

"
advanced_rstheader2 = "
Executor: Internal classes for managing symbolic graph execution
----------------------------------------------------------------------

"
disqus_id = paste0("   :disqus_identifier: ",'advanced')
writeLines(c(advanced_rstheader, formatFuncsToctree(kv_funcs,func_names,func_desc), 
             advanced_rstheader2, formatFuncsToctree(exec_funcs,func_names,func_desc),
             rstfooter, disqus_id), paste0(rst_dir,"advanced.rst"))

monitoring_rstheader = "Monitoring
==============================

Callback: Functions to track various status during an epoch
-------------------------------------------------------------

"
monitoring_rstheader2 = "
Profiler: Running time and memory consumption of MXNet models
---------------------------------------------------------------------------

"
disqus_id = paste0("   :disqus_identifier: ",'monitoring')
writeLines(c(monitoring_rstheader, formatFuncsToctree(callback_funcs,func_names,func_desc), 
             monitoring_rstheader2, formatFuncsToctree(profiler_funcs,func_names,func_desc),
             rstfooter, disqus_id), paste0(rst_dir,"monitoring.rst"))


# Create allfuncs.rst:
allfunctions_rstheader = "Complete Library  
===============================================

List of all MXNet functions available in R
-----------------------------------------------

.. toctree::
   :titlesonly:
"
disqus_id = paste0("   :disqus_identifier: ",'allfunctions')
writeLines(c(allfunctions_rstheader, paste0("   ",func_names),
             rstfooter, disqus_id), paste0(rst_dir,"allfunctions.rst"))








###################
### STOP HERE! ####
### OLD ###########
###################


### TODO:
- ndarray symbol subsections:
 - load/save/load.json

rstcontents = '.. raw:: html

   <div class="mx-api">

.. role:: hidden
    :class: hidden-section
' # header for each RST file.

# Create SIMPLE index.rst:
indexrst_file = paste0(rstfiles_dir,'index.rst')
indexrst_header = " R API 
=============

.. toctree::
   :maxdepth: 1
"
# writeLines(c(indexrst_header, paste(rst_subdir,func_names,sep='')), indexrst_file)


writeLines(c(indexrst_header, paste("   ", rst_subdir, func_names,sep='')), indexrst_file) # Need 3 spaces before each function name.


    # Post-process markdown file:
    contents = readLines(mdfile)
    contents = formatMath(contents,rdcontents)
    contents = formatNote(contents)
    firstline = contents[1]
    contents[1] = paste("# `",func_name,"`",sep='') # replace with only function name.
    firstline_desc = trimws(gsub("#.*`.*`:","", firstline)) # ignore function name and only keep description.
    i = 2
    while ((nchar(trimws(contents[i])) > 0) && (!grepl("Description",contents[i]))) {
        firstline_desc = paste(firstline_desc, trimws(contents[i]))
        contents[i] = ""
        i = i+1
    }
    subfuncname = strsplit(func_name, "\\.")[[1]][length(strsplit(func_name, "\\.")[[1]])]
    firstline_desc = trimws(gsub(paste(subfuncname,":",sep=''), "", firstline_desc)) # remove function name if it appears again.
    if (nchar(firstline_desc) > 0) {
        desctag_index = min(which(contents == "## Description")) # Find official description
        if (desctag_index != Inf) { # description tag exists
            j = desctag_index+1
            while ((j < length(contents)) && (nchar(contents[j]) == 0)) {
                j = j+1
            } # Official Description begins here:
            official_desc = contents[j]
            first_desc_index = j
            j = j+1
            while ((j < length(contents)) && (nchar(contents[j]) != 0)) {
                official_desc = paste(official_desc,"\n",contents[j],sep='')
                j = j+1
            }
            last_desc_index = j+1
            if (!grepl(tolower(firstline_desc), tolower(official_desc), fixed=TRUE)) {
                # Description from first line is not redundant, so should be added to official description.
                # official_desc = paste("\n\n",firstline_desc,"\n",official_desc,"\n\n", sep='')
                first_char = toupper(substr(firstline_desc,1,1)) # Capitalize first character of description
                firstline_desc = paste(first_char,substr(firstline_desc,2,nchar(firstline_desc)), sep='')
                last_char = substr(firstline_desc,nchar(firstline_desc),nchar(firstline_desc))
                if (!grepl('[[:punct:]]', last_char)) { # missing punctuation at the end.
                    firstline_desc = paste(firstline_desc,".",sep="")
                }
                contents = c(contents[1:(first_desc_index-1)],firstline_desc,contents[first_desc_index:length(contents)])
            }
        }
    }
    writeLines(contents, mdfile)

 