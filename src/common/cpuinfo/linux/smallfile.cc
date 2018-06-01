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

namespace mxnet {
namespace common {
namespace cpuinfo {


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

} // cpuinfo
} // common
} // mxnet

#endif