#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
	#include <unistd.h>
#endif
#ifdef __ANDROID__
	#include <android/log.h>
	#define CPUINFO_LOG_TAG "cpuinfo"
#endif

#include "./log.h"

#ifndef CPUINFO_LOG_TO_STDIO
	#ifdef __ANDROID__
		#define CPUINFO_LOG_TO_STDIO 0
	#else
		#define CPUINFO_LOG_TO_STDIO 1
	#endif
#endif

void cpuinfo_log_fatal(const char* format, ...) {
	va_list args;
	va_start(args, format);

	#if defined(__ANDROID__) && !CPUINFO_LOG_TO_STDIO
		__android_log_vprint(ANDROID_LOG_FATAL, CPUINFO_LOG_TAG, format, args);
	#elif defined(__ANDROID__) || defined(_WIN32)
		fprintf(stderr, "Fatal error: ");
		vfprintf(stderr, format, args);
		fprintf(stderr, "\n");
		fflush(stderr);
	#else
		dprintf(STDERR_FILENO, "Error: ");
		vdprintf(STDERR_FILENO, format, args);
		dprintf(STDERR_FILENO, "\n");
	#endif

	va_end(args);
	abort();
}

#if CPUINFO_LOG_LEVEL >= CPUINFO_LOG_ERROR
	void cpuinfo_log_error(const char* format, ...) {
		va_list args;
		va_start(args, format);

		#if defined(__ANDROID__) && !CPUINFO_LOG_TO_STDIO
			__android_log_vprint(ANDROID_LOG_ERROR, CPUINFO_LOG_TAG, format, args);
		#elif defined(__ANDROID__) || defined(_WIN32)
			fprintf(stderr, "Error: ");
			vfprintf(stderr, format, args);
			fprintf(stderr, "\n");
			fflush(stderr);
		#else
			dprintf(STDERR_FILENO, "Error: ");
			vdprintf(STDERR_FILENO, format, args);
			dprintf(STDERR_FILENO, "\n");
		#endif

		va_end(args);
	}
#endif

#if CPUINFO_LOG_LEVEL >= CPUINFO_LOG_WARNING
	void cpuinfo_log_warning(const char* format, ...) {
		va_list args;
		va_start(args, format);

		#if defined(__ANDROID__) && !CPUINFO_LOG_TO_STDIO
			__android_log_vprint(ANDROID_LOG_WARN, CPUINFO_LOG_TAG, format, args);
		#elif defined(__ANDROID__) || defined(_WIN32)
			fprintf(stderr, "Warning: ");
			vfprintf(stderr, format, args);
			fprintf(stderr, "\n");
			fflush(stderr);
		#else
			dprintf(STDERR_FILENO, "Warning: ");
			vdprintf(STDERR_FILENO, format, args);
			dprintf(STDERR_FILENO, "\n");
		#endif

		va_end(args);
	}
#endif

#if CPUINFO_LOG_LEVEL >= CPUINFO_LOG_INFO
	void cpuinfo_log_info(const char* format, ...) {
		va_list args;
		va_start(args, format);

		#if defined(__ANDROID__) && !CPUINFO_LOG_TO_STDIO
			__android_log_vprint(ANDROID_LOG_INFO, CPUINFO_LOG_TAG, format, args);
		#elif defined(__ANDROID__) || defined(_WIN32)
			printf("Note: ");
			vprintf(format, args);
			printf("\n");
			fflush(stdout);
		#else
			vdprintf(STDOUT_FILENO, format, args);
			dprintf(STDOUT_FILENO, "\n");
		#endif

		va_end(args);
	}
#endif

#if CPUINFO_LOG_LEVEL >= CPUINFO_LOG_DEBUG
	void cpuinfo_log_debug(const char* format, ...) {
		va_list args;
		va_start(args, format);

		#if defined(__ANDROID__) && !CPUINFO_LOG_TO_STDIO
			__android_log_vprint(ANDROID_LOG_DEBUG, CPUINFO_LOG_TAG, format, args);
		#elif defined(__ANDROID__) || defined(_WIN32)
			printf("Debug: ");
			vprintf(format, args);
			printf("\n");
			fflush(stdout);
		#else
			vdprintf(STDOUT_FILENO, format, args);
			dprintf(STDOUT_FILENO, "\n");
		#endif

		va_end(args);
	}
#endif
