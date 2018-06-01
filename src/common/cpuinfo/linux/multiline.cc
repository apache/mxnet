//Copyright (c) 2017-2018 Facebook Inc.
//Copyright (C) 2012-2017 Georgia Institute of Technology
//Copyright (C) 2010-2012 Marat Dukhan
//
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//this list of conditions and the following disclaimer in the documentation
//and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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


bool cpuinfo_linux_parse_multiline_file(const char* filename, size_t buffer_size, cpuinfo_line_callback callback, void* context)
{
	int file = -1;
	bool status = false;
	char* buffer = (char*) alloca(buffer_size);

	/* Only used for error reporting */
	size_t position = 0;
	uint64_t line_number = 1;
	const char* buffer_end = &buffer[buffer_size];
	char* data_start = buffer;
	ssize_t bytes_read;

	file = open(filename, O_RDONLY);
	if (file == -1) {
		cpuinfo_log_info("failed to open %s: %s", filename, strerror(errno));
		goto cleanup;
	}


	do {
		bytes_read = read(file, data_start, (size_t) (buffer_end - data_start));

		const char* data_end = data_start + (size_t) bytes_read;
		const char* line_start = buffer;

		if (bytes_read < 0) {
			cpuinfo_log_info("failed to read file %s at position %zu: %s",
				filename, position, strerror(errno));
			goto cleanup;
		}

		position += (size_t) bytes_read;

		if (bytes_read == 0) {
			/* No more data in the file: process the remaining text in the buffer as a single entry */
			const char* line_end = data_end;
			if (!callback(line_start, line_end, context, line_number)) {
				goto cleanup;
			}
		} else {
			const char* line_end;
			do {
				/* Find the end of the entry, as indicated by newline character ('\n') */
				for (line_end = line_start; line_end != data_end; line_end++) {
					if (*line_end == '\n') {
						break;
					}
				}

				/*
				 * If we located separator at the end of the entry, parse it.
				 * Otherwise, there may be more data at the end; read the file once again.
				 */
				if (line_end != data_end) {
					if (!callback(line_start, line_end, context, line_number++)) {
						goto cleanup;
					}
					line_start = line_end + 1;
				}
			} while (line_end != data_end);

			/* Move remaining partial line data at the end to the beginning of the buffer */
			const size_t line_length = (size_t) (line_end - line_start);
			memmove(buffer, line_start, line_length);
			data_start = &buffer[line_length];
		}
	} while (bytes_read != 0);

	/* Commit */
	status = true;

cleanup:
	if (file != -1) {
		close(file);
		file = -1;
	}
	return status;
}

#endif
