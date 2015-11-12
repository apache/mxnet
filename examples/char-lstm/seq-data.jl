# Simple data provider that load text
using MXNet

const UNKNOWN_CHAR = Char(0)

function build_vocabulary(corpus_fn::AbstractString, vocab_fn::AbstractString; max_vocab=10000)
  if isfile(vocab_fn)
    info("Vocabulary already exists, reusing $vocab_fn...")
    vocab = open(corpus_fn) do io
      Dict([w[1] => i for (i,w) in enumerate(eachline(io))])
    end
  else
    # count symbol frequency
    dict = Dict{Char,Int}()
    open(corpus_fn) do io
      for line in eachline(io)
        for c in line
          dict[c] = get(dict, c, 0) + 1
        end
      end
    end

    vocab = sort(collect(dict), by=x->-x.second)
    vocab = vocab[1:min(max_vocab,length(vocab))]
    open(vocab_fn, "w") do io
      for x in vocab
        println(io, x.first)
      end
    end

    vocab = Dict([x.first => i for (i,x) in enumerate(vocab)])
  end
  vocab[UNKNOWN_CHAR] = 0
  return vocab
end

build_vocabulary("input.txt", "vocab.txt")
