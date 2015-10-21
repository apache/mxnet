using MXNet
using Lexicon

config = Config(md_permalink = false, mathjax = true)

doc = Lexicon.metadata(MXNet.mx)
for mod in [:ndarray, :symbol]
  save("api/$mod.md", MIME("text/md"), filter(doc, files=["$mod.jl"]), config)
end

