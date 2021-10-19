/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file extension.h
 * \brief some extension of expressions,
 *  used to support something beyond elementwise op
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXTENSION_H_
#define MSHADOW_EXTENSION_H_
#include "./expr_engine-inl.h"
#include "./extension/broadcast.h"
#include "./extension/unpack_patch2col.h"
#include "./extension/pack_col2patch.h"
#include "./extension/reshape.h"
#include "./extension/swapaxis.h"
#include "./extension/reduceto1d.h"
#include "./extension/spatial_pool.h"
#include "./extension/spatial_unpool.h"
#include "./extension/channel_pool.h"
#include "./extension/channel_unpool.h"
#include "./extension/pad.h"
#include "./extension/crop.h"
#include "./extension/mirror.h"
#include "./extension/concat.h"
#include "./extension/implicit_gemm.h"
#include "./extension/choose.h"
#include "./extension/fill.h"
#include "./extension/one_hot.h"
#include "./extension/slice.h"
#include "./extension/slice_ex.h"
#include "./extension/take.h"
#include "./extension/take_grad.h"
#include "./extension/reduce_with_axis.h"
#include "./extension/broadcast_with_axis.h"
#include "./extension/spatial_upsampling_nearest.h"
#include "./extension/transpose.h"
#include "./extension/flip.h"
#include "./extension/complex.h"
#include "./extension/range.h"
#include "./extension/mask.h"
#endif  // MSHADOW_EXTENSION_H_
