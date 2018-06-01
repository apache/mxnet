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

#include <stdint.h>
#include <stdbool.h>

#include "../cpuinfo.h"
#include "./utils.h"
#include "./api.h"
#include "./cpuid.h"
#include <dmlc/logging.h>


enum topology_type {
	topology_type_invalid = 0,
	topology_type_smt     = 1,
	topology_type_core    = 2,
};

void cpuinfo_x86_detect_topology(
	uint32_t max_base_index,
	uint32_t max_extended_index,
	struct cpuid_regs leaf1,
	struct cpuinfo_x86_topology* topology)
{
	/*
	 * HTT: indicates multi-core/hyper-threading support on this core.
	 * - Intel, AMD: edx[bit 28] in basic info.
	 */
	const bool htt = !!(leaf1.edx & UINT32_C(0x10000000));

	uint32_t apic_id = 0;
	if (htt) {
		apic_id = leaf1.ebx >> 24;
		const uint32_t logical_processors = (leaf1.ebx >> 16) & UINT32_C(0x000000FF);
		if (logical_processors != 0) {
			const uint32_t log2_max_logical_processors = bit_length(logical_processors);
			const uint32_t log2_max_threads_per_core = log2_max_logical_processors - topology->core_bits_length;
			topology->core_bits_offset = log2_max_threads_per_core;
			topology->thread_bits_length = log2_max_threads_per_core;
		}
	}

	/*
	 * x2APIC: indicated support for x2APIC feature.
	 * - Inte: ecx[bit 21] in basic info (reserved bit on AMD CPUs).
	 */
	const bool x2apic = !!(leaf1.ecx & UINT32_C(0x00200000));
	if (x2apic && (max_base_index >= UINT32_C(0xB))) {
		uint32_t level = 0;
		uint32_t type;
		uint32_t total_shift = 0;
		topology->thread_bits_offset = topology->thread_bits_length  = 0;
		topology->core_bits_offset   = topology->core_bits_length = 0;
		do {
			const struct cpuid_regs leafB = cpuidex(UINT32_C(0xB), level);
			type = (leafB.ecx >> 8) & UINT32_C(0x000000FF);
			const uint32_t level_shift = leafB.eax & UINT32_C(0x0000001F);
			const uint32_t x2apic_id   = leafB.edx;
			apic_id = x2apic_id;
			switch (type) {
				case topology_type_invalid:
					break;
				case topology_type_smt:
					cpuinfo_log_debug("x2 level %: APIC ID = %08, "
						"type SMT, shift %, total shift %",
						level, apic_id, level_shift, total_shift);
					topology->thread_bits_offset = total_shift;
					topology->thread_bits_length = level_shift;
					break;
				case topology_type_core:
					cpuinfo_log_debug("x2 level %: APIC ID = %08, "
						"type core, shift %, total shift %",
						level, apic_id, level_shift, total_shift);
					topology->core_bits_offset = total_shift;
					topology->core_bits_length = level_shift;
					break;
				default:
					cpuinfo_log_warning("unexpected topology type % (offset %, length %) "
						"reported in leaf 0x0000000B is ignored", type, total_shift, level_shift);
					break;
			}
			total_shift += level_shift;
			level += 1;
		} while (type != 0);
		cpuinfo_log_debug("x2APIC ID 0x%08, "
			"SMT offset % length %, core offset % length %", apic_id,
			topology->thread_bits_offset, topology->thread_bits_length,
			topology->core_bits_offset, topology->core_bits_length);
	}

	topology->apic_id = apic_id;
}
