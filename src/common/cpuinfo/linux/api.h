#if defined(__linux__)

#pragma once

#define CPUINFO_LINUX_FLAG_PRESENT            UINT32_C(0x00000001)
#define CPUINFO_LINUX_FLAG_POSSIBLE           UINT32_C(0x00000002)
#define CPUINFO_LINUX_MASK_USABLE             UINT32_C(0x00000003)
#define CPUINFO_LINUX_FLAG_MAX_FREQUENCY      UINT32_C(0x00000004)
#define CPUINFO_LINUX_FLAG_MIN_FREQUENCY      UINT32_C(0x00000008)
#define CPUINFO_LINUX_FLAG_SMT_ID             UINT32_C(0x00000010)
#define CPUINFO_LINUX_FLAG_CORE_ID            UINT32_C(0x00000020)
#define CPUINFO_LINUX_FLAG_PACKAGE_ID         UINT32_C(0x00000040)
#define CPUINFO_LINUX_FLAG_APIC_ID            UINT32_C(0x00000080)
#define CPUINFO_LINUX_FLAG_SMT_CLUSTER        UINT32_C(0x00000100)
#define CPUINFO_LINUX_FLAG_CORE_CLUSTER       UINT32_C(0x00000200)
#define CPUINFO_LINUX_FLAG_PACKAGE_CLUSTER    UINT32_C(0x00000400)

#include <stdbool.h>
#include <stdint.h>

#include <stddef.h>
#include "../../cpuinfo.h"
#include "../api.h"


struct cpuinfo_x86_linux_processor {
	uint32_t apic_id;
	uint32_t linux_id;
	uint32_t flags;
};

bool cpuinfo_x86_linux_parse_proc_cpuinfo(
	uint32_t max_processors_count,
	cpuinfo_x86_linux_processor* processors);

typedef bool (*cpuinfo_cpulist_callback)(uint32_t, uint32_t, void*);
bool cpuinfo_linux_parse_cpulist(const char* filename, cpuinfo_cpulist_callback callback, void* context);
typedef bool (*cpuinfo_smallfile_callback)(const char*, const char*, void*);
bool cpuinfo_linux_parse_small_file(const char* filename, size_t buffer_size, cpuinfo_smallfile_callback, void* context);
typedef bool (*cpuinfo_line_callback)(const char*, const char*, void*, uint64_t);
bool cpuinfo_linux_parse_multiline_file(const char* filename, size_t buffer_size, cpuinfo_line_callback, void* context);

uint32_t cpuinfo_linux_get_max_processors_count(void);
uint32_t cpuinfo_linux_get_max_possible_processor(uint32_t max_processors_count);
uint32_t cpuinfo_linux_get_max_present_processor(uint32_t max_processors_count);
uint32_t cpuinfo_linux_get_processor_min_frequency(uint32_t processor);
uint32_t cpuinfo_linux_get_processor_max_frequency(uint32_t processor);
bool cpuinfo_linux_get_processor_package_id(uint32_t processor, uint32_t package_id[1]);
bool cpuinfo_linux_get_processor_core_id(uint32_t processor, uint32_t core_id[1]);

bool cpuinfo_linux_detect_possible_processors(uint32_t max_processors_count,
																							uint32_t* processor0_flags, uint32_t processor_struct_size, uint32_t possible_flag);
bool cpuinfo_linux_detect_present_processors(uint32_t max_processors_count,
																						 uint32_t* processor0_flags, uint32_t processor_struct_size, uint32_t present_flag);
typedef bool (*cpuinfo_siblings_callback)(uint32_t, uint32_t, uint32_t, void*);

extern const struct cpuinfo_processor** cpuinfo_linux_cpu_to_processor_map;
extern const struct cpuinfo_core** cpuinfo_linux_cpu_to_core_map;

#endif