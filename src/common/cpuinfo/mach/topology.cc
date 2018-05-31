#if defined(__MACH__) && defined(__APPLE__)

#include <string.h>
#include <alloca.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/sysctl.h>

#include "../log.h"
#include "./api.h"


struct cpuinfo_mach_topology cpuinfo_mach_detect_topology(void) {
	int cores = 1;
	size_t sizeof_cores = sizeof(cores);
	if (sysctlbyname("hw.physicalcpu_max", &cores, &sizeof_cores, NULL, 0) != 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.physicalcpu_max\") failed: %s", strerror(errno));
	} else if (cores <= 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.physicalcpu_max\") returned invalid value %d", cores);
		cores = 1;
	}

	int threads = 1;
	size_t sizeof_threads = sizeof(threads);
	if (sysctlbyname("hw.logicalcpu_max", &threads, &sizeof_threads, NULL, 0) != 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.logicalcpu_max\") failed: %s", strerror(errno));
	} else if (threads <= 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.logicalcpu_max\") returned invalid value %d", threads);
		threads = cores;
	}

	int packages = 1;
	size_t sizeof_packages = sizeof(packages);
	if (sysctlbyname("hw.packages", &packages, &sizeof_packages, NULL, 0) != 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.packages\") failed: %s", strerror(errno));
	} else if (packages <= 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.packages\") returned invalid value %d", packages);
		packages = 1;
	}

	cpuinfo_log_info("mach topology: packages = %d, cores = %d, threads = %d", packages, (int) cores, (int) threads);
	struct cpuinfo_mach_topology topology = {
		.packages = (uint32_t) packages,
		.cores = (uint32_t) cores,
		.threads = (uint32_t) threads
	};

	size_t cacheconfig_size = 0;
	if (sysctlbyname("hw.cacheconfig", NULL, &cacheconfig_size, NULL, 0) != 0) {
		cpuinfo_log_error("sysctlbyname(\"hw.cacheconfig\") failed: %s", strerror(errno));
	} else {
		uint64_t* cacheconfig = (uint64_t*) alloca(cacheconfig_size);
		if (sysctlbyname("hw.cacheconfig", cacheconfig, &cacheconfig_size, NULL, 0) != 0) {
			cpuinfo_log_error("sysctlbyname(\"hw.cacheconfig\") failed: %s", strerror(errno));
		} else {
			size_t cache_configs = cacheconfig_size / sizeof(uint64_t);
			cpuinfo_log_debug("mach hw.cacheconfig count: %zu", cache_configs);
			if (cache_configs > CPUINFO_MACH_MAX_CACHE_LEVELS) {
				cache_configs = CPUINFO_MACH_MAX_CACHE_LEVELS;
			}
			for (size_t i = 0; i < cache_configs; i++) {
				cpuinfo_log_debug("mach hw.cacheconfig[%zu]: %"PRIu64, i, cacheconfig[i]);
				topology.threads_per_cache[i] = cacheconfig[i];
			}
		}
	}
	return topology;
}

#endif
