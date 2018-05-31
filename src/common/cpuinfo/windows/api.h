#ifdef _WIN32

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <windows.h>

#include "../../cpuinfo.h"
#include "../api.h"

struct cpuinfo_arm_linux_processor {
	/**
	 * Minimum processor ID on the package which includes this logical processor.
	 * This value can serve as an ID for the cluster of logical processors: it is the
	 * same for all logical processors on the same package.
	 */
	uint32_t package_leader_id;
	/**
	 * Minimum processor ID on the core which includes this logical processor.
	 * This value can serve as an ID for the cluster of logical processors: it is the
	 * same for all logical processors on the same package.
	 */
	/**
	 * Number of logical processors in the package.
	 */
	uint32_t package_processor_count;
	/**
	 * Maximum frequency, in kHZ.
	 * The value is parsed from /sys/devices/system/cpu/cpu<N>/cpufreq/cpuinfo_max_freq
	 * If failed to read or parse the file, the value is 0.
	 */
	uint32_t max_frequency;
	/**
	 * Minimum frequency, in kHZ.
	 * The value is parsed from /sys/devices/system/cpu/cpu<N>/cpufreq/cpuinfo_min_freq
	 * If failed to read or parse the file, the value is 0.
	 */
	uint32_t min_frequency;
	/** Linux processor ID */
	uint32_t system_processor_id;
	uint32_t flags;
};

#endif