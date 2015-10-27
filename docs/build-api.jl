# extract API docs

function extract_doc(output_filename::AbstractString, input_filenames::AbstractString...)
  src_dir = joinpath(Pkg.dir("MXNet"), "src")
  api_dir = joinpath(Pkg.dir("MXNet"), "docs", "api")

  mkpath(api_dir)
  open(joinpath(api_dir, output_filename), "w") do io
    for in_fn in input_filenames
      for doc in eachmatch(r"^#=doc\s*$(.*?)^=#\s*$"ms, readall(joinpath(src_dir, in_fn)))
        println(io, doc.captures[1], "\n\n")
      end
    end
  end
end

extract_doc("ndarray.rst", "ndarray.jl")
