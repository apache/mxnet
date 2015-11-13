include(joinpath(dirname(@__FILE__), "config.jl"))
include(joinpath(dirname(@__FILE__), "seq-data.jl"))

using StatsBase
using MXNet

# load vocabulary
vocab   = build_vocabulary(INPUT_FILE, VOCAB_FILE)

# prepare data provider
jl_data = Pair[(symbol(NAME, "_data_$t") => zeros(mx.MX_float, (length(vocab), BATCH_SIZE_SMP)))
               for t = 1:SEQ_LENGTH]
jl_c    = Pair[(symbol(NAME, "_l$(l)_init_c") => zeros(mx.MX_float, (DIM_HIDDEN, BATCH_SIZE_SMP)))
               for l = 1:LSTM_N_LAYER]
jl_h    = Pair[(symbol(NAME, "_l$(l)_init_h") => zeros(mx.MX_float, (DIM_HIDDEN, BATCH_SIZE_SMP)))
               for l = 1:LSTM_N_LAYER]

# the first input in the sequence
jl_data_start = jl_data[1].second
jl_data_start[char_idx(vocab, SAMPLE_START),:] = 1

# load model
model = mx.load_checkpoint(CKPOINT_PREFIX, N_EPOCH, mx.FeedForward)

# prepare outputs
Base.zero(::Type{Char}) = Char(0)
output_samples = zeros(Char, (SAMPLE_LENGTH, BATCH_SIZE_SMP))
output_samples[1, :] = SAMPLE_START

# build inverse vocabulary for convenience
inv_vocab = Dict([v => k for (k,v) in vocab])

# do prediction and sampling step by step
for t = 2:SAMPLE_LENGTH-1
  data    = mx.ArrayDataProvider(jl_data ∪ jl_c ∪ jl_h)
  outputs = mx.predict(model, data)

  # we will only use the first output to do sampling
  outputs = outputs[1]

  jl_data_start[:] = 0
  for i = 1:BATCH_SIZE_SMP
    prob = WeightVec(outputs[:, i])
    k    = sample(prob)
    output_samples[t, i] = inv_vocab[k]
    jl_data_start[k, i]  = 1
  end
end

output_texts = [join(output_samples[:,i]) for i = 1:BATCH_SIZE_SMP]
output_texts = [replace(x, UNKNOWN_CHAR, '?') for x in output_texts]

for (i, text) in enumerate(output_texts)
  println("## Sample $i")
  println(text)
  println()
end
