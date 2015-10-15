# Internal function to check if name end with suffix
mx.util.str.endswith <- function(name, suffix) {
  slen <- nchar(suffix)
  nlen <- nchar(name)
  if (slen > nlen) return (FALSE)
  nsuf <- substr(name, nlen - slen + 1, nlen)
  return (nsuf == suffix)
}

mx.util.str.startswith <- function(name, prefix) {
  slen <- nchar(prefix)
  nlen <- nchar(name)
  if (slen > nlen) return (FALSE)
  npre <- substr(name, 1, slen)
  return (npre == prefix)
}

# filter out null, keep the names
mx.util.filter.null <- function(lst) {
  lst[!sapply(lst, is.null)]
}
