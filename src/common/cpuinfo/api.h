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

#include <stdbool.h>
#include <stdint.h>

#include "../cpuinfo.h"

#ifdef _WIN32
#include <windows.h>
#endif


struct cpuid_regs {
	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;
};


struct cpuinfo_x86_topology {
	uint32_t apic_id;
	uint32_t thread_bits_offset;
	uint32_t thread_bits_length;
	uint32_t core_bits_offset;
	uint32_t core_bits_length;
};

struct cpuinfo_x86_processor {
	uint32_t cpuid;
	struct cpuinfo_x86_topology topology;
};

void cpuinfo_x86_init_processor(struct cpuinfo_x86_processor* processor);

void cpuinfo_x86_detect_topology(
	uint32_t max_base_index,
	uint32_t max_extended_index,
	struct cpuid_regs leaf1,
	struct cpuinfo_x86_topology* topology);


extern bool cpuinfo_is_initialized;
extern struct cpuinfo_processor* cpuinfo_processors;
extern struct cpuinfo_core* cpuinfo_cores;
extern uint32_t cpuinfo_cores_count;

void cpuinfo_x86_mach_init(void);
void cpuinfo_x86_linux_init(void);
#ifdef _WIN32
BOOL CALLBACK cpuinfo_x86_windows_init(PINIT_ONCE init_once, PVOID parameter, PVOID* context);
#endif

typedef void (*cpuinfo_processor_callback)(uint32_t);

