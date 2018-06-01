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

	struct cpuinfo_mach_topology topology = {
		.cores = (uint32_t) cores,
	};

	return topology;
}

#endif