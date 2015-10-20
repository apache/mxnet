using MXNet
using Lexicon

config = Config(md_permalink = false, mathjax = true)

index = save("api/MXNet.md", mx, config)

