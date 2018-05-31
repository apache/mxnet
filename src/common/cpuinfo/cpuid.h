#pragma once
#include <stdint.h>

#if defined(__GNUC__)
	#include <cpuid.h>
#elif defined(_MSC_VER)
	#include <intrin.h>
#endif

#include "api.h"


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

/*
 * This instruction may be not supported by Native Client validator,
 * make sure it doesn't appear in the binary
 */
#ifndef __native_client__
	static inline uint64_t xgetbv(uint32_t ext_ctrl_reg) {
		#ifdef _MSC_VER
			return (uint64_t)_xgetbv((unsigned int)ext_ctrl_reg);
		#else
			uint32_t lo, hi;
			__asm__(".byte 0x0F, 0x01, 0xD0" : "=a" (lo), "=d" (hi) : "c" (ext_ctrl_reg));
			return ((uint64_t) hi << 32) | (uint64_t) lo;
		#endif
	}
#endif
