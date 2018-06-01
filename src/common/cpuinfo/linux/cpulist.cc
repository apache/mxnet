#if defined(__linux__)

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>

#include "./api.h"
#include "../log.h"


/*
 * Size, in chars, of the on-stack buffer used for parsing cpu lists.
 * This is also the limit on the length of a single entry
 * (<cpu-number> or <cpu-number-start>-<cpu-number-end>)
 * in the cpu list.
 */
#define BUFFER_SIZE 256


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

inline static const char* parse_number(const char* string, const char* end, uint32_t number_ptr[1]) {
	uint32_t number = 0;
	while (string != end) {
		const uint32_t digit = (uint32_t) (*string) - (uint32_t) '0';
		if (digit >= 10) {
			break;
		}
		number = number * UINT32_C(10) + digit;
		string += 1;
	}
	*number_ptr = number;
	return string;
}

inline static bool parse_entry(const char* entry_start, const char* entry_end, cpuinfo_cpulist_callback callback, void* context) {
	/* Skip whitespace at the beginning of an entry */
	for (; entry_start != entry_end; entry_start++) {
		if (!is_whitespace(*entry_start)) {
			break;
		}
	}
	/* Skip whitespace at the end of an entry */
	for (; entry_end != entry_start; entry_end--) {
		if (!is_whitespace(entry_end[-1])) {
			break;
		}
	}

	const size_t entry_length = (size_t) (entry_end - entry_start);
	if (entry_length == 0) {
		cpuinfo_log_warning("unexpected zero-length cpu list entry ignored");
		return false;
	}

	uint32_t first_cpu, last_cpu;

	const char* number_end = parse_number(entry_start, entry_end, &first_cpu);
	if (number_end == entry_start) {
		/* Failed to parse the number; ignore the entry */
		cpuinfo_log_warning("invalid character '%c' in the cpu list entry \"%.*s\": entry is ignored",
			entry_start[0], (int) entry_length, entry_start);
		return false;
	} else if (number_end == entry_end) {
		/* Completely parsed the entry */
		return callback(first_cpu, first_cpu + 1, context);
	}

	/* Parse the second part of the entry */
	if (*number_end != '-') {
		cpuinfo_log_warning("invalid character '%c' in the cpu list entry \"%.*s\": entry is ignored",
			*number_end, (int) entry_length, entry_start);
		return false;
	}

	const char* number_start = number_end + 1;
	number_end = parse_number(number_start, entry_end, &last_cpu);
	if (number_end == number_start) {
		/* Failed to parse the second number; ignore the entry */
		cpuinfo_log_warning("invalid character '%c' in the cpu list entry \"%.*s\": entry is ignored",
			*number_start, (int) entry_length, entry_start);
		return false;
	}

	if (number_end != entry_end) {
		/* Partially parsed the entry; ignore unparsed characters and continue with the parsed part */
		cpuinfo_log_warning("ignored invalid characters \"%.*s\" at the end of cpu list entry \"%.*s\"",
			(int) (entry_end - number_end), number_start, (int) entry_length, entry_start);
	}

	if (last_cpu < first_cpu) {
		cpuinfo_log_warning("ignored cpu list entry \"%.*s\": invalid range %-%",
			(int) entry_length, entry_start, first_cpu, last_cpu);
		return false;
	}

	/* Parsed both parts of the entry; update CPU set */
	return callback(first_cpu, last_cpu + 1, context);
}

bool cpuinfo_linux_parse_cpulist(const char* filename, cpuinfo_cpulist_callback callback, void* context) {
	bool status = true;
	int file = -1;
	char buffer[BUFFER_SIZE];

	size_t position = 0;
	const char* buffer_end = &buffer[BUFFER_SIZE];
	char* data_start = buffer;
	ssize_t bytes_read;

	file = open(filename, O_RDONLY);
	if (file == -1) {
		cpuinfo_log_info("failed to open %s: %s", filename, strerror(errno));
		status = false;
		goto cleanup;
	}

	do {

		bytes_read = read(file, data_start, (size_t) (buffer_end - data_start));
		if (bytes_read < 0) {
			cpuinfo_log_info("failed to read file %s at position %zu: %s", filename, position, strerror(errno));
			status = false;
			goto cleanup;
		}

		position += (size_t) bytes_read;
		const char* data_end = data_start + (size_t) bytes_read;
		const char* entry_start = buffer;

		if (bytes_read == 0) {
			/* No more data in the file: process the remaining text in the buffer as a single entry */
			const char* entry_end = data_end;
			const bool entry_status = parse_entry(entry_start, entry_end, callback, context);
			status &= entry_status;
		} else {
			const char* entry_end;
			do {
				/* Find the end of the entry, as indicated by a comma (',') */
				for (entry_end = entry_start; entry_end != data_end; entry_end++) {
					if (*entry_end == ',') {
						break;
					}
				}

				/*
				 * If we located separator at the end of the entry, parse it.
				 * Otherwise, there may be more data at the end; read the file once again.
				 */
				if (entry_end != data_end) {
					const bool entry_status = parse_entry(entry_start, entry_end, callback, context);
					status &= entry_status;
					entry_start = entry_end + 1;
				}
			} while (entry_end != data_end);

			/* Move remaining partial entry data at the end to the beginning of the buffer */
			const size_t entry_length = (size_t) (entry_end - entry_start);
			memmove(buffer, entry_start, entry_length);
			data_start = &buffer[entry_length];
		}
	} while (bytes_read != 0);

	cleanup:
		if (file != -1) {
			close(file);
			file = -1;
		}
	return status;
}

#endif