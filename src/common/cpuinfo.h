//Copyright (c) 2017-2018 Facebook Inc.
//Copyright (C) 2012-2017 Georgia Institute of Technology
//Copyright (C) 2010-2012 Marat Dukhan
//
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//this list of conditions and the following disclaimer in the documentation
//and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <stdint.h>

/* Identify architecture and define corresponding macro */

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
#define CPUINFO_ARCH_X86 1
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#define CPUINFO_ARCH_X86_64 1
#endif


#if CPUINFO_ARCH_X86 && defined(_MSC_VER)
#define CPUINFO_ABI __cdecl
#elif CPUINFO_ARCH_X86 && defined(__GNUC__)
#define CPUINFO_ABI __attribute__((__cdecl__))
#else
#define CPUINFO_ABI
#endif

/* Define other architecture-specific macros as 0 */

#ifndef CPUINFO_ARCH_X86
#define CPUINFO_ARCH_X86 0
#endif

#ifndef CPUINFO_ARCH_X86_64
#define CPUINFO_ARCH_X86_64 0
#endif

namespace mxnet {
namespace common {
namespace cpuinfo {

bool CPUINFO_ABI cpuinfo_initialize(void);
uint32_t CPUINFO_ABI cpuinfo_get_cores_count(void);

} // cpuinfo
} // common
} // mxnet

