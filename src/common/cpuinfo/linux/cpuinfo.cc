#if defined(__linux__)

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "./api.h"
#include "../log.h"

/*
 * Size, in chars, of the on-stack buffer used for parsing lines of /proc/cpuinfo.
 * This is also the limit on the length of a single line.
 */
#define BUFFER_SIZE 2048


static uint32_t parse_processor_number(
	const char* processor_start,
	const char* processor_end)
{
	const size_t processor_length = (size_t) (processor_end - processor_start);

	if (processor_length == 0) {
		cpuinfo_log_warning("Processor number in /proc/cpuinfo is ignored: string is empty");
		return 0;
	}

	uint32_t processor_number = 0;
	for (const char* digit_ptr = processor_start; digit_ptr != processor_end; digit_ptr++) {
		const uint32_t digit = (uint32_t) (*digit_ptr - '0');
		if (digit > 10) {
			cpuinfo_log_warning("non-decimal suffix %.*s in /proc/cpuinfo processor number is ignored",
				(int) (processor_end - digit_ptr), digit_ptr);
			break;
		}

		processor_number = processor_number * 10 + digit;
	}

	return processor_number;
}

/*
 * Decode APIC ID reported by Linux kernel for x86/x86-64 architecture.
 * Example of APIC ID reported in /proc/cpuinfo:
 *
 *		apicid		: 2
 */
static void parse_apic_id(
	const char* apic_start,
	const char* apic_end,
	struct cpuinfo_x86_linux_processor processor[1])
{
	uint32_t apic_id = 0;
	for (const char* digit_ptr = apic_start; digit_ptr != apic_end; digit_ptr++) {
		const uint32_t digit = *digit_ptr - '0';
		if (digit >= 10) {
			cpuinfo_log_warning("APIC ID %.*s in /proc/cpuinfo is ignored due to unexpected non-digit character '%c' at offset %zu",
				(int) (apic_end - apic_start), apic_start,
				*digit_ptr, (size_t) (digit_ptr - apic_start));
			return;
		}

		apic_id = apic_id * 10 + digit;
	}

	processor->apic_id = apic_id;
	processor->flags |= CPUINFO_LINUX_FLAG_APIC_ID;
}

struct proc_cpuinfo_parser_state {
	uint32_t processor_index;
	uint32_t max_processors_count;
	struct cpuinfo_x86_linux_processor* processors;
	struct cpuinfo_x86_linux_processor dummy_processor;
};

/*
 *	Decode a single line of /proc/cpuinfo information.
 *	Lines have format <words-with-spaces>[ ]*:[ ]<space-separated words>
 */
static bool parse_line(
	const char* line_start,
	const char* line_end,
	struct proc_cpuinfo_parser_state state[1],
	uint64_t line_number)
{
	/* Empty line. Skip. */
	if (line_start == line_end) {
		return true;
	}

	/* Search for ':' on the line. */
	const char* separator = line_start;
	for (; separator != line_end; separator++) {
		if (*separator == ':') {
			break;
		}
	}
	/* Skip line if no ':' separator was found. */
	if (separator == line_end) {
		cpuinfo_log_warning("Line %.*s in /proc/cpuinfo is ignored: key/value separator ':' not found",
			(int) (line_end - line_start), line_start);
		return true;
	}

	/* Skip trailing spaces in key part. */
	const char* key_end = separator;
	for (; key_end != line_start; key_end--) {
		if (key_end[-1] != ' ' && key_end[-1] != '\t') {
			break;
		}
	}
	/* Skip line if key contains nothing but spaces. */
	if (key_end == line_start) {
		cpuinfo_log_warning("Line %.*s in /proc/cpuinfo is ignored: key contains only spaces",
			(int) (line_end - line_start), line_start);
		return true;
	}

	/* Skip leading spaces in value part. */
	const char* value_start = separator + 1;
	for (; value_start != line_end; value_start++) {
		if (*value_start != ' ') {
			break;
		}
	}
	/* Value part contains nothing but spaces. Skip line. */
	if (value_start == line_end) {
		cpuinfo_log_warning("Line %.*s in /proc/cpuinfo is ignored: value contains only spaces",
			(int) (line_end - line_start), line_start);
		return true;
	}

	/* Skip trailing spaces in value part (if any) */
	const char* value_end = line_end;
	for (; value_end != value_start; value_end--) {
		if (value_end[-1] != ' ') {
			break;
		}
	}

	const uint32_t processor_index      = state->processor_index;
	const uint32_t max_processors_count = state->max_processors_count;
	struct cpuinfo_x86_linux_processor* processors = state->processors;
	struct cpuinfo_x86_linux_processor* processor  = &state->dummy_processor;
	if (processor_index < max_processors_count) {
		processor = &processors[processor_index];
	}

	const size_t key_length = key_end - line_start;
	switch (key_length) {
		case 6:
			if (memcmp(line_start, "apicid", key_length) == 0) {
				parse_apic_id(value_start, value_end, processor);
			} else {
				goto unknown;
			}
			break;
		case 9:
			if (memcmp(line_start, "processor", key_length) == 0) {
				const uint32_t new_processor_index = parse_processor_number(value_start, value_end);
				if (new_processor_index < processor_index) {
					/* Strange: decreasing processor number */
					cpuinfo_log_warning(
						"unexpectedly low processor number % following processor % in /proc/cpuinfo",
						new_processor_index, processor_index);
				} else if (new_processor_index > processor_index + 1) {
					/* Strange, but common: skipped processor $(processor_index + 1) */
					cpuinfo_log_info(
						"unexpectedly high processor number % following processor % in /proc/cpuinfo",
						new_processor_index, processor_index);
				}
				if (new_processor_index >= max_processors_count) {
					/* Log and ignore processor */
					cpuinfo_log_warning("processor % in /proc/cpuinfo is ignored: index exceeds system limit %",
						new_processor_index, max_processors_count - 1);
				}
				state->processor_index = new_processor_index;
				return true;
			} else {
				goto unknown;
			}
			break;
		default:
		unknown:
			cpuinfo_log_debug("unknown /proc/cpuinfo key: %.*s", (int) key_length, line_start);

	}
	return true;
}

bool cpuinfo_x86_linux_parse_proc_cpuinfo(
	uint32_t max_processors_count,
	cpuinfo_x86_linux_processor* processors)
{
	struct proc_cpuinfo_parser_state state = {
		.processor_index = 0,
		.max_processors_count = max_processors_count,
		.processors = processors,
	};
	return cpuinfo_linux_parse_multiline_file("/proc/cpuinfo", BUFFER_SIZE,
		(cpuinfo_line_callback) parse_line, &state);
}

#endif