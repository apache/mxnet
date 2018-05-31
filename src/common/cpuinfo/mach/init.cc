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