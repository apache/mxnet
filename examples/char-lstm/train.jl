include(joinpath(dirname(@__FILE__), "config.jl"))
include(joinpath(dirname(@__FILE__), "lstm.jl"))
include(joinpath(dirname(@__FILE__), "seq-data.jl"))

# build vocabulary
vocab   = build_vocabulary(INPUT_FILE, VOCAB_FILE)
n_class = length(vocab)

# define LSTM
lstm = LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_class, dropout=DROPOUT, name=NAME)

# load data
text_all  = readall(INPUT_FILE)
len_train = round(Int, length(text_all)*DATA_TR_RATIO)
text_tr   = text_all[1:len_train]
text_val  = text_all[len_train+1:end]

data_tr   = CharSeqProvider(text_tr, BATCH_SIZE, SEQ_LENGTH, vocab, NAME,
                            LSTM_N_LAYER, DIM_HIDDEN)
data_val  = CharSeqProvider(text_val, BATCH_SIZE, SEQ_LENGTH, vocab, NAME,
                            LSTM_N_LAYER, DIM_HIDDEN)

# set up training
if USE_GPU
  context = [mx.gpu(i) for i = 0:N_GPU-1]
else
  context = [mx.cpu()]
end

model = mx.FeedForward(lstm, context=context)
optimizer = mx.SGD(lr=LEARNING_RATE, momentum=MOMENTUM,
                   weight_decay=WEIGHT_DECAY, grad_clip=CLIP_GRADIENT)

mx.fit(model, optimizer, data_tr, eval_data=data_val, n_epoch=N_EPOCH,
       callbacks=[mx.speedometer()], eval_metric=NLL())
