# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

const DROPOUT        = 0
const BATCH_SIZE     = 32
const SEQ_LENGTH     = 32
const DIM_HIDDEN     = 256
const DIM_EMBED      = 256
const LSTM_N_LAYER   = 2
const N_EPOCH        = 21
const BASE_LR        = 0.01
const WEIGHT_DECAY   = 0.00001
const CLIP_GRADIENT  = 1
const NAME           = :ptb
const N_GPU          = 1
const USE_GPU        = true
const DATA_TR_RATIO  = 0.9
const CKPOINT_PREFIX = joinpath(@__DIR__, "checkpoints/$NAME")

const BATCH_SIZE_SMP = 10
const SAMPLE_LENGTH  = 100
const SAMPLE_START   = 'a'

const UNKNOWN_CHAR   = Char(0)
const INPUT_FILE     = joinpath(@__DIR__, "input.txt")
const VOCAB_FILE     = joinpath(@__DIR__, "vocab.dat")
