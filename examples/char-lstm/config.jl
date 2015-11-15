const DROPOUT       = 0
const BATCH_SIZE    = 32
const SEQ_LENGTH    = 32
const DIM_HIDDEN    = 256
const DIM_EMBED     = 256
const LSTM_N_LAYER  = 2
const N_EPOCH       = 21
const BASE_LR       = 0.01
const WEIGHT_DECAY  = 0.00001
const CLIP_GRADIENT = 1
const NAME          = :ptb
const N_GPU         = 4
const USE_GPU       = true
const DATA_TR_RATIO = 0.9
const CKPOINT_PREFIX = joinpath(dirname(@__FILE__), "checkpoints/$NAME")

const BATCH_SIZE_SMP= 10
const SAMPLE_LENGTH = 100
const SAMPLE_START  = 'a'

const UNKNOWN_CHAR  = Char(0)
const INPUT_FILE    = joinpath(dirname(@__FILE__), "input.txt")
const VOCAB_FILE    = joinpath(dirname(@__FILE__), "vocab.dat")

