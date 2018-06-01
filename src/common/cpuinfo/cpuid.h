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
#include <stdint.h>

#if defined(__GNUC__)
	#include <cpuid.h>
#elif defined(_MSC_VER)
	#include <intrin.h>
#endif

#include "api.h"

namespace mxnet {
namespace common {
namespace cpuinfo {


#if defined(__GNUC__) || defined(_MSC_VER)
	static inline struct cpuid_regs cpuid(uint32_t eax) {
		#if CPUINFO_MOCK
			uint32_t regs_array[4];
			cpuinfo_mock_get_cpuid(eax, regs_array);
			return (struct cpuid_regs) {
				.eax = regs_array[0],
				.ebx = regs_array[1],
				.ecx = regs_array[2],
				.edx = regs_array[3],
			};
		#else
			struct cpuid_regs regs;
			#if defined(__GNUC__)
				__cpuid(eax, regs.eax, regs.ebx, regs.ecx, regs.edx);
			#else
				int regs_array[4];
				__cpuid(regs_array, (int) eax);
				regs.eax = regs_array[0];
				regs.ebx = regs_array[1];
				regs.ecx = regs_array[2];
				regs.edx = regs_array[3];
			#endif
			return regs;
		#endif
	}

	static inline struct cpuid_regs cpuidex(uint32_t eax, uint32_t ecx) {
		#if CPUINFO_MOCK
			uint32_t regs_array[4];
			cpuinfo_mock_get_cpuidex(eax, ecx, regs_array);
			return (struct cpuid_regs) {
				.eax = regs_array[0],
				.ebx = regs_array[1],
				.ecx = regs_array[2],
				.edx = regs_array[3],
			};
		#else
			struct cpuid_regs regs;
			#if defined(__GNUC__)
				__cpuid_count(eax, ecx, regs.eax, regs.ebx, regs.ecx, regs.edx);
			#else
				int regs_array[4];
				__cpuidex(regs_array, (int) eax, (int) ecx);
				regs.eax = regs_array[0];
				regs.ebx = regs_array[1];
				regs.ecx = regs_array[2];
				regs.edx = regs_array[3];
			#endif
			return regs;
		#endif
	}
#endif

} // cpuinfo
} // common
} // mxnet