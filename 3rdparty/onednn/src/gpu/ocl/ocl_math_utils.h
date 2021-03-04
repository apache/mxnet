/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_OCL_MATH_UTILS_H
#define GPU_OCL_OCL_MATH_UTILS_H

// f32 <-> bf16 conversion.
#if DT_BF16 || SRC_DT_BF16 || WEI_DT_BF16 || DST_DT_BF16 || BIA_DT_BF16 \
        || A_DT_BF16 || B_DT_BF16 || C_DT_BF16 || SUM_DT_BF16 \
        || POST_OP_USING_BF16
ushort __attribute__((overloadable)) cvt_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

ushort2 __attribute__((overloadable)) cvt_f32_to_bf16(float2 f) {
    ushort2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort4 __attribute__((overloadable)) cvt_f32_to_bf16(float4 f) {
    ushort4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort8 __attribute__((overloadable)) cvt_f32_to_bf16(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

float __attribute__((overloadable)) cvt_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

float2 __attribute__((overloadable)) cvt_bf16_to_f32(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float4 __attribute__((overloadable)) cvt_bf16_to_f32(ushort4 b) {
    float4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float8 __attribute__((overloadable)) cvt_bf16_to_f32(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}
#endif

int __attribute__((overloadable)) idot4(char4 a, char4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(uchar4 a, uchar4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(char4 a, uchar4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(uchar4 a, char4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(int a, int b, int c) {
    return idot4(as_char4(a), as_char4(b), c);
}

int __attribute__((overloadable)) idot4(uint a, int b, int c) {
    return idot4(as_uchar4(a), as_char4(b), c);
}

#define DECLARE_BLOCK_READ(suffix, func, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix(const addr_space p_type *p) { \
        return func(p); \
    }

#define DECLARE_BLOCK_READ_EMU(suffix, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix##_emu(const addr_space p_type *p) { \
        data_type ret; \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            ((p_type *)&ret)[i] = p[idx]; \
            idx += get_max_sub_group_size(); \
        } \
        return ret; \
    }

#define DECLARE_BLOCK_WRITE(suffix, func, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix(addr_space p_type *p, data_type data) { \
        func(p, data); \
    }

#define DECLARE_BLOCK_WRITE_EMU(suffix, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix##_emu(addr_space p_type *p, data_type data) { \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            p[idx] = ((p_type *)&data)[i]; \
            p += get_max_sub_group_size(); \
        } \
    }

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __global, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __global, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __global, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __global, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __global, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __global, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __global, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __global, uint)

#ifdef cl_intel_subgroups_char
void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(const __global uchar *p);
#endif

// Emulation for cl_intel_subgroup_local_block_io. These functions are not
// defined under ifndef/endif because some kernels rely on the emulation
// functions in case when pointers are not properly aligned for the native
// extensions.
DECLARE_BLOCK_READ_EMU(, uint, __local, uint)
DECLARE_BLOCK_READ_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_READ_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_READ_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(, uint, __local, uint)
DECLARE_BLOCK_WRITE_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_WRITE_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_WRITE_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(_us, ushort, __local, ushort)

#ifdef cl_intel_subgroup_local_block_io

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __local, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __local, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __local, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(
        _us, intel_sub_group_block_write_us, ushort, __local, ushort)

#else

DECLARE_BLOCK_READ(, block_read_emu, uint, __local, uint)
DECLARE_BLOCK_READ(2, block_read2_emu, uint2, __local, uint)
DECLARE_BLOCK_READ(4, block_read4_emu, uint4, __local, uint)
DECLARE_BLOCK_READ(8, block_read8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, block_write_emu, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, block_write2_emu, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, block_write4_emu, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, block_write8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(_us, block_write_us_emu, ushort, __local, ushort)

#endif

// Integer matrix-matrix multiplication: ACC += A * B
// A is (m x (4 * K))
// B is ((4 * K) x sub_group_size)
#define DECLARE_MMAD(name, K, m, a_type, b_type, acc_type) \
    acc_type __attribute__((overloadable)) \
            name(a_type A_vectors, b_type B_vectors, acc_type acc) { \
        for (uint i = 0; i < (m); ++i) { \
            for (uint j = 0; j < (K); ++j) \
                acc[i] = idot4(sub_group_broadcast(A_vectors[i], j), \
                        B_vectors[j], acc[i]); \
        } \
        return acc; \
    }

DECLARE_MMAD(mmad8x4, 8, 4, uint4, int8, int4)
DECLARE_MMAD(mmad8x4, 8, 4, int4, int8, int4)
DECLARE_MMAD(mmad8x8, 8, 8, uint8, int8, int8)
DECLARE_MMAD(mmad8x8, 8, 8, int8, int8, int8)

#endif
