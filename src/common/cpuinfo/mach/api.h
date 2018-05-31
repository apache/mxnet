#if defined(__MACH__) && defined(__APPLE__)

#pragma once

#include <stdint.h>

#define CPUINFO_MACH_MAX_CACHE_LEVELS 8


struct cpuinfo_mach_topology {
	uint32_t packages;
	uint32_t cores;
	uint32_t threads;
	uint32_t threads_per_cache[CPUINFO_MACH_MAX_CACHE_LEVELS];
};


struct cpuinfo_mach_topology cpuinfo_mach_detect_topology(void);

#endif