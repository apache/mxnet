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

#if defined(__MACH__) && defined(__APPLE__)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../../cpuinfo.h"
#include "../api.h"
#include "./api.h"
#include "../log.h"

static inline uint32_t max(uint32_t a, uint32_t b) {
	return a > b ? a : b;
}

static inline uint32_t bit_mask(uint32_t bits) {
	return (UINT32_C(1) << bits) - UINT32_C(1);
}

void cpuinfo_x86_mach_init(void) {
	struct cpuinfo_mach_topology mach_topology = cpuinfo_mach_detect_topology();
	cpuinfo_cores_count = mach_topology.cores;
	__sync_synchronize();
	cpuinfo_is_initialized = true;

}

#endif