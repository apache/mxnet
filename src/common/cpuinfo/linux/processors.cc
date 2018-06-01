#if defined(__linux__)

#define _GNU_SOURCE 1
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sched.h>
#include "./api.h"
#include "../log.h"

#define KERNEL_MAX_FILENAME "/sys/devices/system/cpu/kernel_max"
#define KERNEL_MAX_FILESIZE 32
#define POSSIBLE_CPULIST_FILENAME "/sys/devices/system/cpu/possible"
#define PRESENT_CPULIST_FILENAME "/sys/devices/system/cpu/present"


inline static const char* parse_number(const char* start, const char* end, uint32_t number_ptr[1]) {
	uint32_t number = 0;
	const char* parsed = start;
	for (; parsed != end; parsed++) {
		const uint32_t digit = (uint32_t) (uint8_t) (*parsed) - (uint32_t) '0';
		if (digit >= 10) {
			break;
		}
		number = number * UINT32_C(10) + digit;
	}
	*number_ptr = number;
	return parsed;
}

/* Locale-independent */
inline static bool is_whitespace(char c) {
	switch (c) {
		case ' ':
		case '\t':
		case '\n':
		case '\r':
			return true;
		default:
			return false;
	}
}

static const uint32_t default_max_processors_count = CPU_SETSIZE;

static bool uint32_parser(const char* text_start, const char* text_end, void* context) {
	if (text_start == text_end) {
		cpuinfo_log_error("failed to parse file %s: file is empty", KERNEL_MAX_FILENAME);
		return false;
	}

	uint32_t kernel_max = 0;
	const char* parsed_end = parse_number(text_start, text_end, &kernel_max);
	if (parsed_end == text_start) {
		cpuinfo_log_error("failed to parse file %s: \"%.*s\" is not an unsigned number",
			KERNEL_MAX_FILENAME, (int) (text_end - text_start), text_start);
		return false;
	} else {
		for (const char* char_ptr = parsed_end; char_ptr != text_end; char_ptr++) {
			if (!is_whitespace(*char_ptr)) {
				cpuinfo_log_warning("non-whitespace characters \"%.*s\" following number in file %s are ignored",
					(int) (text_end - char_ptr), char_ptr, KERNEL_MAX_FILENAME);
				break;
			}
		}
	}

	uint32_t* kernel_max_ptr = (uint32_t*) context;
	*kernel_max_ptr = kernel_max;
	return true;
}

uint32_t cpuinfo_linux_get_max_processors_count(void) {
	uint32_t kernel_max;
	if (cpuinfo_linux_parse_small_file(KERNEL_MAX_FILENAME, KERNEL_MAX_FILESIZE, uint32_parser, &kernel_max)) {
		cpuinfo_log_debug("parsed kernel_max value of % from %s", kernel_max, KERNEL_MAX_FILENAME);

		if (kernel_max >= default_max_processors_count) {
			cpuinfo_log_warning("kernel_max value of % parsed from %s exceeds platform-default limit %",
				kernel_max, KERNEL_MAX_FILENAME, default_max_processors_count - 1);
		}

		return kernel_max + 1;
	} else {
		cpuinfo_log_warning("using platform-default max processors count = %", default_max_processors_count);
		return default_max_processors_count;
	}
}

static bool max_processor_number_parser(uint32_t processor_list_start, uint32_t processor_list_end, void* context) {
	uint32_t* processor_number_ptr = (uint32_t*) context;
	const uint32_t processor_list_last = processor_list_end - 1;
	if (*processor_number_ptr < processor_list_last) {
		*processor_number_ptr = processor_list_last;
	}
	return true;
}

uint32_t cpuinfo_linux_get_max_possible_processor(uint32_t max_processors_count) {
	uint32_t max_possible_processor = 0;
	if (!cpuinfo_linux_parse_cpulist(POSSIBLE_CPULIST_FILENAME, max_processor_number_parser, &max_possible_processor)) {
		cpuinfo_log_error("failed to parse the list of possible procesors in %s", POSSIBLE_CPULIST_FILENAME);
		return max_processors_count;
	}
	if (max_possible_processor >= max_processors_count) {
		cpuinfo_log_warning(
			"maximum possible processor number % exceeds system limit %: truncating to the latter",
			max_possible_processor, max_processors_count - 1);
		max_possible_processor = max_processors_count - 1;
	}
	return max_possible_processor;
}

uint32_t cpuinfo_linux_get_max_present_processor(uint32_t max_processors_count) {
	uint32_t max_present_processor = 0;
	if (!cpuinfo_linux_parse_cpulist(PRESENT_CPULIST_FILENAME, max_processor_number_parser, &max_present_processor)) {
		cpuinfo_log_error("failed to parse the list of present procesors in %s", PRESENT_CPULIST_FILENAME);
		return max_processors_count;
	}
	if (max_present_processor >= max_processors_count) {
		cpuinfo_log_warning(
			"maximum present processor number % exceeds system limit %: truncating to the latter",
			max_present_processor, max_processors_count - 1);
		max_present_processor = max_processors_count - 1;
	}
	return max_present_processor;
}

struct detect_processors_context {
	uint32_t max_processors_count;
	uint32_t* processor0_flags;
	uint32_t processor_struct_size;
	uint32_t detected_flag;
};

static bool detect_processor_parser(uint32_t processor_list_start, uint32_t processor_list_end, void* context) {
	const uint32_t max_processors_count   = ((struct detect_processors_context*) context)->max_processors_count;
	const uint32_t* processor0_flags      = ((struct detect_processors_context*) context)->processor0_flags;
	const uint32_t processor_struct_size  = ((struct detect_processors_context*) context)->processor_struct_size;
	const uint32_t detected_flag          = ((struct detect_processors_context*) context)->detected_flag;

	for (uint32_t processor = processor_list_start; processor < processor_list_end; processor++) {
		if (processor >= max_processors_count) {
			break;
		}
		*((uint32_t*) ((void*) processor0_flags + processor_struct_size * processor)) |= detected_flag;
	}
	return true;
}

bool cpuinfo_linux_detect_possible_processors(uint32_t max_processors_count,
	uint32_t* processor0_flags, uint32_t processor_struct_size, uint32_t possible_flag)
{
	struct detect_processors_context context = {
		.max_processors_count = max_processors_count,
		.processor0_flags = processor0_flags,
		.processor_struct_size = processor_struct_size,
		.detected_flag = possible_flag,
	};
	if (cpuinfo_linux_parse_cpulist(POSSIBLE_CPULIST_FILENAME, detect_processor_parser, &context)) {
		return true;
	} else {
		cpuinfo_log_warning("failed to parse the list of possible procesors in %s", POSSIBLE_CPULIST_FILENAME);
		return false;
	}
}

bool cpuinfo_linux_detect_present_processors(uint32_t max_processors_count,
	uint32_t* processor0_flags, uint32_t processor_struct_size, uint32_t present_flag)
{
	struct detect_processors_context context = {
		.max_processors_count = max_processors_count,
		.processor0_flags = processor0_flags,
		.processor_struct_size = processor_struct_size,
		.detected_flag = present_flag,
	};
	if (cpuinfo_linux_parse_cpulist(PRESENT_CPULIST_FILENAME, detect_processor_parser, &context)) {
		return true;
	} else {
		cpuinfo_log_warning("failed to parse the list of present procesors in %s", PRESENT_CPULIST_FILENAME);
		return false;
	}
}

#endif

