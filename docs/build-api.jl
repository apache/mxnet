# extract API docs
using MXNet

const SRC_DIR = joinpath(Pkg.dir("MXNet"), "src")
const API_DIR = joinpath(Pkg.dir("MXNet"), "docs", "api")

function extract_doc(output_filename::AbstractString, input_filenames::AbstractString...)

  mkpath(API_DIR)
  open(joinpath(API_DIR, output_filename), "w") do io
    for in_fn in input_filenames
      for doc in eachmatch(r"^#=doc\s*$(.*?)^=#\s*$"ms, readall(joinpath(SRC_DIR, in_fn)))
        println(io, doc.captures[1], "\n\n")
      end
    end
  end
end

function sort_api_names(names)
  names = collect(names)
  names_pub = filter(x -> !startswith(string(x), '_'), names)
  names_pri = filter(x -> startswith(string(x), '_'), names)
  return (sort(names_pub), sort(names_pri))
end

function embed_mxnet_api(output_filename::AbstractString, key::AbstractString, generator::Function)
  output_filename = joinpath(API_DIR, output_filename)
  contents = readall(output_filename)
  open(output_filename, "w") do io
    docs = generator(gen_docs=true)
    function gen_doc(fname)
      doc = replace(docs[fname], r"^"m, "   ")
      """
      .. function:: $fname(...)

      $doc

      """
    end

    names_pub, names_pri = sort_api_names(keys(docs))
    docs_pub = join(map(gen_doc, names_pub), "\n\n")
    docs_pri = join(map(gen_doc, names_pri), "\n\n")
    docstrings = """
    Public APIs
    ^^^^^^^^^^^
    """ * docs_pub

    docstrings *= """

    Internal APIs
    ^^^^^^^^^^^^^

    .. note::

       Document and signatures for internal API functions might be incomplete.

    """ * docs_pri

    key = mx.format(mx.DOC_EMBED_ANCHOR, key)
    println(io, replace(contents, key, docstrings))
  end
end

extract_doc("ndarray.rst", "ndarray.jl")
embed_mxnet_api("ndarray.rst", "ndarray", mx._import_ndarray_functions)
