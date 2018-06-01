#pragma once
#ifndef CPUINFO_H
#define CPUINFO_H

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <stdint.h>

/* Identify architecture and define corresponding macro */

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
#define CPUINFO_ARCH_X86 1
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#define CPUINFO_ARCH_X86_64 1
#endif


#if CPUINFO_ARCH_X86 && defined(_MSC_VER)
#define CPUINFO_ABI __cdecl
#elif CPUINFO_ARCH_X86 && defined(__GNUC__)
#define CPUINFO_ABI __attribute__((__cdecl__))
#else
#define CPUINFO_ABI
#endif

/* Define other architecture-specific macros as 0 */

#ifndef CPUINFO_ARCH_X86
#define CPUINFO_ARCH_X86 0
#endif

#ifndef CPUINFO_ARCH_X86_64
#define CPUINFO_ARCH_X86_64 0
#endif

bool CPUINFO_ABI cpuinfo_initialize(void);

uint32_t CPUINFO_ABI cpuinfo_get_cores_count(void);

#endif /* CPUINFO_H */
