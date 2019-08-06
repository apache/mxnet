# Helper functions for Rdoc2SphinxDoc.R
# Used to convert Rd documentation to Sphinx RST documentation:

getTagText <- function(tag_pattern, tag_index, alltag_indices, contentlines) {
    # Always assumes tag_index is correctly specified.
    if (!grepl(tag_pattern,contentlines[tag_index],fixed=TRUE)) {
        stop("tag_pattern not present at specied tag_index line of contentlines")
    }
    tagtext = trimws(gsub(tag_pattern,"",contentlines[tag_index],fixed = TRUE))
    i = tag_index + 1
    higher_tagindices = alltag_indices[which(alltag_indices > tag_index)]
    if (length(higher_tagindices) > 0) {
        next_tagindex = min(higher_tagindices)
    } else {
        next_tagindex = length(contentlines)
    }
    while (i < next_tagindex) {
        tagtext = c(tagtext, contentlines[i])
        i = i + 1
    }
    i = length(tagtext)
    while (nchar(trimws(tagtext[i])) == 0) { # remove empty strings at beginning/end of lines.
        tagtext = tagtext[1:(i-1)]
        i = length(tagtext)
    }
    if (trimws(tagtext[i]) == '}') {
        tagtext = tagtext[1:(i-1)]
    } else if (substr(tagtext[i],nchar(tagtext[i]),nchar(tagtext[i])) == '}') {
        tagtext[i] = substr(tagtext[i],1,nchar(tagtext[i])-1)
    }
    while (nchar(trimws(tagtext[1])) == 0) {
        tagtext = tagtext[2:length(tagtext)]
    }
    return(tagtext)
}

punctuate <- function(string) {
    # Adds period to end of string if no end punctuation exists
    string = trimws(string)
    if (nchar(string) == 0) {
        return(string)
    }
    last_char = substr(string,nchar(string),nchar(string))
    if ((!grepl('[[:punct:]]', last_char)) || grepl("'|)|}|\\]",last_char)) { # missing punctuation at the end.
        string = paste0(string,".  ")
    }
    return(string)
}

underbarTitle <- function(string, symbol = "-", extra_newline = TRUE) {
    # Produces sufficiently long underline (with SYMBOL) for string in RST format and returns underlined title as vector:
    if (extra_newline) {
        return(c(string, paste0(rep(symbol,nchar(string)*2),collapse=''),""))
    } else {
        return(c(string, paste0(rep(symbol,nchar(string)*2),collapse='')))
    }
}

createlink <-function(linkstr, srcsite) {
    colonsplits = strsplit(linkstr,split=":")[[1]]
    if ((length(colonsplits) > 1) && (grepl("L[0-9]",colonsplits[length(colonsplits)]))) {
        # Line number exists in link:
        linenum = paste0("#",trimws(colonsplits[length(colonsplits)]))
        linkstr = paste0(paste0(colonsplits[1:(length(colonsplits)-1)], collapse=""),linenum)
    }
    # TODO: process extra backslashes in linkstr?
    url = paste0(srcsite,linkstr)
    url = gsub("\\","",url,fixed=T)
    # html_link = paste0('<a href="',url,'">',linkstr,'</a>') # formatted.
    html_link = url # unformatted. Let Sphinx / javascript style determine formatting.
    return(url)
}

argsTableRow <- function(argname, argdesc,LHS_WIDTH, RHS_WIDTH) {
    # Formats single row (cell) of RST arguments table:
    argname = trimws(argname)
    oldargdesc = trimws(argdesc)
    argdesc = character(0)
    for (i in 1:length(oldargdesc)) {
        if (nchar(oldargdesc[i])+2 > RHS_WIDTH) {
            # Need to split descriptions into chunks to fit into cell:
            words = strsplit(oldargdesc[i],split=" ")[[1]]
            j = 1
            line_j = ""
            while (j <= length(words)) {
                if (nchar(words[j])+2 > RHS_WIDTH) {
                    stop(paste0("RHS_WIDTH=",RHS_WIDTH," is too small. Needs to be at least >=",nchar(words[j])+3), 
                         " for: ",words[j])
                }
                if (nchar(line_j)+nchar(words[j])+1+2 <= RHS_WIDTH) {
                    line_j = trimws(paste(line_j, words[j]))
                    j = j+1
                }
                if ((nchar(line_j)+nchar(words[j])+1+2 > RHS_WIDTH) || (j >= length(words))) {
                    argdesc = c(argdesc, line_j)
                    line_j = ""
                }
            }
        } else {
            argdesc = c(argdesc, oldargdesc[i])
        }
    }
    cell = character(0)
    for (i in 1:max(length(argname),length(argdesc))) {
        if (i <= length(argname)) {
            if (nchar(argname[i])+2 > LHS_WIDTH) {
                stop(paste0("LHS_WIDTH=",LHS_WIDTH," is too small. Needs to be at least >=",nchar(argname[i])+2,
                    " for: ",argname[i]))
            }
            nameline = paste0(" ",argname[i], paste0(rep(" ",LHS_WIDTH - nchar(argname[i])-1),collapse=""))
        } else {
            nameline = paste0(rep(" ", LHS_WIDTH),collapse="")
        }
        if (i <= length(argdesc)) {
            descline = paste0(" ",argdesc[i], paste0(rep(" ",RHS_WIDTH - nchar(argdesc[i])-1),collapse=""))
        } else {
            descline = paste0(rep(" ", RHS_WIDTH),collapse="")
        }
        cell = c(cell, paste0("|",nameline,"|",descline,"|"))
    }
    # print(cell)
    return(cell)
}

formatRSTindentation <- function(rstcontents) {
    # Formats "Examples::" in RST file as well as other listed blocks prefaced by "::"
    exindexes = c(grep('Examples::',rstcontents),grep('Example::',rstcontents),
                  grep('examples::',rstcontents),grep('example::',rstcontents),
                  grep('Ex::',rstcontents),grep('ex::',rstcontents))
    # rstcontents[first_exindex] = ".. examples::"
    if (length(exindexes) > 0) {
        todelete = numeric(0)
        for (exindex in exindexes) {
            index = exindex + 1
            exstring = ""
            while ((index < length(rstcontents)) && (!grepl("Defined in",rstcontents[index])) && 
                (!grepl("Value",rstcontents[index])) && (!grepl("Arguments",rstcontents[index])) &&
                (!grepl("Usage",rstcontents[index])) && (!grepl(".. Note::",rstcontents[index],fixed=T)) &&
                (!grepl("Example::",rstcontents[index],fixed=T))
                   ) {
                exstring = paste0(exstring, rstcontents[index],"\n")
                if (rstcontents[index] =='::')  {
                    todelete = c(todelete, index)
                }
                rstcontents[index] = paste0("\t ",trimws(rstcontents[index]))
                index = index + 1
            }
        }
        if (length(todelete) > 0) {
            rstcontents = rstcontents[-todelete]
        }
    }
    # Now repeat similar processing for other blocks that have prefix "::"
    # remove unnecessary "\\" that occur in these blocks.
    blkindexes = grep('::',rstcontents)
    ignoreindexes = c(exindexes, grep('math::',rstcontents), grep('note::',rstcontents),
                      grep('raw::',rstcontents), grep('html::',rstcontents), grep('role::',rstcontents),
                      grep('class::',rstcontents), grep('disqus::',rstcontents), grep('code::',rstcontents))
    blkindexes = blkindexes[which(!(blkindexes %in% ignoreindexes))]
    if (length(blkindexes) > 0) {
        for (blkindex in blkindexes) {
            index = blkindex + 1
            while ((index < length(rstcontents)) && (nchar(trimws(rstcontents[index])) == 0)) {
                index = index + 1 # skip blank lines.
            }
            blkstring = ""
            num_blank = 0 # number of blank lines in a row encountered.
            while ((index < length(rstcontents)) && (num_blank <= 3) && (!grepl("::$",rstcontents[index])) &&
                   (!grepl("^Defined in",rstcontents[index])) && (!grepl("^Value",rstcontents[index])) &&
                   (!grepl("^Arguments",rstcontents[index])) && (!grepl("^Usage",rstcontents[index])) && 
                   (!grepl(".. Note::",rstcontents[index],fixed=T)) && (!grepl("Example::",rstcontents[index],fixed=T))
                  ) {
                if (nchar(trimws(rstcontents[index]))==0) {
                    num_blank = num_blank + 1
                } else {
                    num_blank = 0
                }
                blkstring = paste0(blkstring, rstcontents[index],"\n")
                blkstring = gsub("\\","",blkstring,fixed=T) # remove unnecessary "\\" that occur in these blocks.
                rstcontents[index] = paste0("\t ",trimws(rstcontents[index]))
                index = index + 1
            }
        }
    }
    # Format .. note:: blocks:
    note_indices = grep(".. note::",rstcontents, fixed=T)
    toremove = numeric(0)
    for (index in note_indices) {
        note_string = rstcontents[index]
        i = index+1
        while ((i < length(rstcontents)) && (nchar(trimws(rstcontents[i])) > 0)) {
            note_string = paste(note_string, rstcontents[i])
            toremove = c(toremove,i)
            i = i+1
        }
        rstcontents[index] = note_string
    }
    if (length(toremove) > 0) {
        rstcontents = rstcontents[-toremove]
    }
    rstcontents = gsub("Example::","**Example**::",rstcontents, fixed=T)
    rstcontents = gsub("Examples::","**Example**::",rstcontents, fixed=T)
    return(rstcontents)
}

formatInlineTags <- function(rstcontents) {
    rstcontents = gsub("\\\\code\\{(.*?)\\}","``\\1``", rstcontents) # format \code{} tags
    rstcontents = gsub("\\\\method\\{(.*?)\\}\\{(.*?)\\}","\\1.\\2", rstcontents) # format \method{funcname}{class} tags.
    rstcontents = gsub("\\\\method\\{(.*?)\\}","\\1", rstcontents) # format \method{funcname} tags.
    return(rstcontents)
}

fixBulletDoc <- function(rstcontents) { 
    # Ensures bulleted list (lines preceded by '-') are indented to same level.
    # TODO: this is a problem in underlying documentation. Should not be need once underyling problem is fixed.
    indices = grep("[a-zA-Z]:$",trimws(rstcontents))
    for (index in indices) {
        num_bullets = 0
        i = index + 1
        if (nchar(trimws(rstcontents[i])) == 0) {
            i = i+1 # allow one blank line
        }
        bullet1_index = i
        while ((i < length(rstcontents)) && (substr(trimws(rstcontents[i]),1,1) == '-')) {
            i = i+1
            num_bullets = num_bullets+1
        }
        if (num_bullets > 1) { # ensure they are indendented together:
            i = bullet1_index
            while ((i < length(rstcontents)) && (substr(trimws(rstcontents[i]),1,1) == '-')) {
                rstcontents[i] = paste0("\t",trimws(rstcontents[i]))
                i = i+1
            }
        }
    }
    return(rstcontents) 
}

formatFuncsToctree <- function(funcstoformat,func_names,func_desc, includetoctree=TRUE,subdir="") {
    # Creates table of function names and description + links in hidden toctree
    if (length(funcstoformat) == 0) { return("") }
    max_name_length = max(as.vector(sapply(funcstoformat, nchar)))
    max_desc_length = max(as.vector(sapply(func_desc[func_names %in% funcstoformat], nchar)))
    name_width = max_name_length*2 + 14 # width of table cell for function names
    desc_width = max_desc_length + 5
    hbar = paste0(paste0(rep("=",name_width),collapse=""),"  ",
                       paste0(rep("=",desc_width),collapse=""))
    formatted = hbar
    for (i in 1:length(funcstoformat)) {
        func = funcstoformat[i]
        if (!(func %in% func_names)) {
            stop(paste("Trying to format unknown function:",func))
        }
        desc = func_desc[which(func_names == func)]
        desc = paste0(desc, collapse=" ")
        newrow = paste0(":doc:`",func," <./",subdir,func,">`")
        if (nchar(desc) > 0) {
            desc = strsplit(desc, split="\n")[[1]]
            newrow = paste0(newrow, paste0(rep(" ", name_width+2-nchar(newrow)),collapse=""), trimws(desc[1]))
            if (length(desc) > 1) {
                for (j in 2:length(desc)) {
                    if (grepl("^[[:upper:]]",trimws(desc[j]))) {
                        if (!grepl("[[:punct:]]$", trimws(desc[j-1]))) { # need to add punctuation to previous row.
                            newrow[length(newrow)] = paste0(newrow[length(newrow)],".  ")
                        }
                    }
                    newrow = c(newrow, paste0(paste0(rep(" ", name_width+2),collapse=""), trimws(desc[j])))
                }
            }
        }
        formatted = c(formatted, newrow)
    }
    formatted = c(formatted,hbar)
    if (includetoctree) {
        toctree_head = "
.. toctree::
   :titlesonly:
   :hidden:
   "
        return(c(formatted,toctree_head,paste0("   ",funcstoformat)))
    } else {
        return(formatted)
    }
}


