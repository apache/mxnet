#if defined(__linux__)

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "./api.h"
#include "../log.h"


bool cpuinfo_linux_parse_small_file(const char* filename, size_t buffer_size, cpuinfo_smallfile_callback callback, void* context) {
	int file = -1;
	bool status = false;
	char* buffer = (char*) alloca(buffer_size);

	size_t buffer_position = 0;
	ssize_t bytes_read;
	file = open(filename, O_RDONLY);
	if (file == -1) {
		cpuinfo_log_info("failed to open %s: %s", filename, strerror(errno));
		goto cleanup;
	}

	do {
		bytes_read = read(file, &buffer[buffer_position], buffer_size - buffer_position);
		if (bytes_read < 0) {
			cpuinfo_log_info("failed to read file %s at position %zu: %s", filename, buffer_position, strerror(errno));
			goto cleanup;
		}
		buffer_position += (size_t) bytes_read;
		if (buffer_position >= buffer_size) {
			cpuinfo_log_error("failed to read file %s: insufficient buffer of size %zu", filename, buffer_size);
			goto cleanup;
		}
	} while (bytes_read != 0);

	status = callback(buffer, &buffer[buffer_position], context);

cleanup:
	if (file != -1) {
		close(file);
		file = -1;
	}
	return status;
}

#endif