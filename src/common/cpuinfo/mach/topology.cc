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