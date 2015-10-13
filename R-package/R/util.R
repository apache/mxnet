# Internal function to check if name end with suffix
mx.util.str.endswith <- function(name, suffix) {
  slen <- nchar(suffix)
  nlen <- nchar(name)
  if (slen > nlen) return (FALSE)
  nsuf <- substr(name, nlen - slen + 1, nlen)
  return (nsuf == suffix)
  # The follow code was not working when suffix is longer than name
  #ptrn = paste0(suffix, "\\b")
  #return(grepl(ptrn, name))
}

# filter out null, keep the names
mx.util.filter.null <- function(lst) {
  lst[!sapply(lst, is.null)]
}
