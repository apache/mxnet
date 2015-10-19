using MXNet
using Lexicon

config = Config(md_permalink = false, mathjax = true)

index = save("api/MXNet.md", MXNet.mx, config)
save("api/index.md", Index([index]), config; md_subheader = :category)

