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

#ifdef _WIN32

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "../cpuinfo.h"
#include "../api.h"
#include "../log.h"
#include <malloc.h>

#define alloca _alloca

#include <windows.h>

namespace mxnet {
namespace common {
namespace cpuinfo {


static inline uint32_t bit_mask(uint32_t bits) {
	return (UINT32_C(1) << bits) - UINT32_C(1);
}

static inline uint32_t low_index_from_kaffinity(KAFFINITY kaffinity) {
	#if defined(_M_X64) || defined(_M_AMD64)
		unsigned long index;
		_BitScanForward64(&index, (unsigned __int64) kaffinity);
		return (uint32_t) index;
	#elif defined(_M_IX86)
		unsigned long index;
		_BitScanForward(&index, (unsigned long) kaffinity);
		return (uint32_t) index;
	#else
		#error Platform-specific implementation required
	#endif
}

BOOL CALLBACK cpuinfo_x86_windows_init(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {

	PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX processor_infos = NULL;

	HANDLE heap = GetProcessHeap();

	struct cpuinfo_x86_processor x86_processor;
	ZeroMemory(&x86_processor, sizeof(x86_processor));
	cpuinfo_x86_init_processor(&x86_processor);

	const uint32_t max_group_count = (uint32_t) GetMaximumProcessorGroupCount();
	cpuinfo_log_debug("detected % processor groups", max_group_count);

	uint32_t processors_count = 0;
	uint32_t* processors_per_group = (uint32_t*) _alloca(max_group_count * sizeof(uint32_t));
	for (uint32_t i = 0; i < max_group_count; i++) {
		processors_per_group[i] = GetMaximumProcessorCount((WORD) i);
		cpuinfo_log_debug("detected % processors in group %",
			processors_per_group[i], i);
		processors_count += processors_per_group[i];
	}

	uint32_t* processors_before_group = (uint32_t*) _alloca(max_group_count * sizeof(uint32_t));
	for (uint32_t i = 0, count = 0; i < max_group_count; i++) {
		processors_before_group[i] = count;
		cpuinfo_log_debug("detected % processors before group %",
			processors_before_group[i], i);
		count += processors_per_group[i];
	}

	DWORD cores_info_size = 0;
	if (GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &cores_info_size) == FALSE) {
		const DWORD last_error = GetLastError();
		if (last_error != ERROR_INSUFFICIENT_BUFFER) {
			cpuinfo_log_error("failed to query size of processor cores information: error %",
				(uint32_t) last_error);
			return TRUE;
		}
	}

	DWORD packages_info_size = 0;
	if (GetLogicalProcessorInformationEx(RelationProcessorPackage, NULL, &packages_info_size) == FALSE) {
		const DWORD last_error = GetLastError();
		if (last_error != ERROR_INSUFFICIENT_BUFFER) {
			cpuinfo_log_error("failed to query size of processor packages information: error %",
				(uint32_t) last_error);
			return TRUE;
		}
	}

	DWORD max_info_size = max(cores_info_size, packages_info_size);

	processor_infos = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) HeapAlloc(heap, 0, max_info_size);

	if (processor_infos == NULL) {
		cpuinfo_log_error("failed to allocate % bytes for logical processor information",
			(uint32_t)max_info_size);
		return TRUE;
	}

	if (GetLogicalProcessorInformationEx(RelationProcessorPackage, processor_infos, &max_info_size) == FALSE) {
		cpuinfo_log_error("failed to query processor packages information: error %",
			(uint32_t)GetLastError());
		return TRUE;
	}

	max_info_size = max(cores_info_size, packages_info_size);
	if (GetLogicalProcessorInformationEx(RelationProcessorCore, processor_infos, &max_info_size) == FALSE) {
		cpuinfo_log_error("failed to query processor cores information: error %",
			(uint32_t)GetLastError());
		return TRUE;
	}


	uint32_t cores_count = 0;
	/* Index (among all cores) of the the first core on the current package */
	uint32_t package_core_start = 0;
	uint32_t current_package_apic_id = 0;
	PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX cores_info_end =
		(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) ((uintptr_t) processor_infos + cores_info_size);
	for (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX core_info = processor_infos;
		core_info < cores_info_end;
		core_info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) ((uintptr_t) core_info + core_info->Size))
	{
		if (core_info->Relationship != RelationProcessorCore) {
			cpuinfo_log_warning("unexpected processor info type (%) for processor core information",
				(uint32_t) core_info->Relationship);
			continue;
		}

		/* We assume that cores and logical processors are reported in APIC order */
		const uint32_t core_id = cores_count++;

	}

	cpuinfo_cores_count = cores_count;

	MemoryBarrier();

	cpuinfo_is_initialized = true;

	return TRUE;
}

} // cpuinfo
} // common
} // mxnet

#endif

