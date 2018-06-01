#if defined(__MACH__) && defined(__APPLE__)

#pragma once

#include <stdint.h>

#define CPUINFO_MACH_MAX_CACHE_LEVELS 8


struct cpuinfo_mach_topology {
	uint32_t cores;
};


struct cpuinfo_mach_topology cpuinfo_mach_detect_topology(void);

#endif