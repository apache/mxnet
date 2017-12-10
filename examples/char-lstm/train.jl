include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "lstm.jl"))
include(joinpath(@__DIR__, "seq-data.jl"))

# build vocabulary
vocab   = build_vocabulary(INPUT_FILE, VOCAB_FILE)
n_class = length(vocab)

#--LSTM
# define LSTM
lstm = LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED,
            n_class, dropout=DROPOUT, name=NAME)
#--/LSTM

#--data
# load data
text_all  = readstring(INPUT_FILE)
len_train = round(Int, length(text_all)*DATA_TR_RATIO)
text_tr   = text_all[1:len_train]
text_val  = text_all[len_train+1:end]

data_tr   = CharSeqProvider(text_tr, BATCH_SIZE, SEQ_LENGTH, vocab, NAME,
                            LSTM_N_LAYER, DIM_HIDDEN)
data_val  = CharSeqProvider(text_val, BATCH_SIZE, SEQ_LENGTH, vocab, NAME,
                            LSTM_N_LAYER, DIM_HIDDEN)
#--/data

# set up training
if USE_GPU
  context = [mx.gpu(i) for i = 0:N_GPU-1]
else
  context = mx.cpu()
end

#--train
model = mx.FeedForward(lstm, context=context)
optimizer = mx.ADAM(lr=BASE_LR, weight_decay=WEIGHT_DECAY, grad_clip=CLIP_GRADIENT)

mx.fit(model, optimizer, data_tr, eval_data=data_val, n_epoch=N_EPOCH,
       initializer=mx.UniformInitializer(0.1),
       callbacks=[mx.speedometer(), mx.do_checkpoint(CKPOINT_PREFIX)], eval_metric=NLL())
#--/train
