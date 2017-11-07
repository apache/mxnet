using Documenter, MXNet

makedocs(
  modules = MXNet,
  doctest = false
)

deploydocs(
  deps = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
  repo = "github.com/dmlc/MXNet.jl.git",
  julia = "0.6",
)
