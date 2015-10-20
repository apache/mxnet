using MXNet
using Lexicon

config = Config(md_permalink = false, mathjax = true)

function save_meta(file :: AbstractString, docs :: Lexicon.Metadata, order = [:source])
  isfile(file) || mkpath(dirname(file))
  open(file, "w") do io
    for (k,v) in Lexicon.EachEntry(docs, order = order)
      name = Lexicon.writeobj(k)
      println(io, "#### $name")
      println(io, v.docs.data)
    end
  end
end

doc = Lexicon.metadata(MXNet.mx)
for mod in [:ndarray, :symbol]
  save("api/$mod.md", MIME("text/md"), filter(doc, files=["$mod.jl"]), config)
end

