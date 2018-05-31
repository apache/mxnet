#pragma once
#ifndef CPUINFO_H
#define CPUINFO_H

#include <stdbool.h>

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

#if defined(__arm__) || defined(_M_ARM)
#define CPUINFO_ARCH_ARM 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define CPUINFO_ARCH_ARM64 1
#endif

#if defined(__PPC64__) || defined(__powerpc64__) || defined(_ARCH_PPC64)
#define CPUINFO_ARCH_PPC64 1
#endif

#if defined(__pnacl__)
#define CPUINFO_ARCH_PNACL 1
#endif

#if defined(EMSCRIPTEN)
#define CPUINFO_ARCH_ASMJS 1
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

#ifndef CPUINFO_ARCH_ARM
#define CPUINFO_ARCH_ARM 0
#endif

#ifndef CPUINFO_ARCH_ARM64
#define CPUINFO_ARCH_ARM64 0
#endif

#ifndef CPUINFO_ARCH_PPC64
#define CPUINFO_ARCH_PPC64 0
#endif

#ifndef CPUINFO_ARCH_PNACL
#define CPUINFO_ARCH_PNACL 0
#endif

#ifndef CPUINFO_ARCH_ASMJS
#define CPUINFO_ARCH_ASMJS 0
#endif

#define CPUINFO_CACHE_UNIFIED          0x00000001
#define CPUINFO_CACHE_INCLUSIVE        0x00000002
#define CPUINFO_CACHE_COMPLEX_INDEXING 0x00000004

#define CPUINFO_PAGE_SIZE_4KB  0x1000
#define CPUINFO_PAGE_SIZE_1MB  0x100000
#define CPUINFO_PAGE_SIZE_2MB  0x200000
#define CPUINFO_PAGE_SIZE_4MB  0x400000
#define CPUINFO_PAGE_SIZE_16MB 0x1000000
#define CPUINFO_PAGE_SIZE_1GB  0x40000000


bool CPUINFO_ABI cpuinfo_initialize(void);

void CPUINFO_ABI cpuinfo_deinitialize(void);

uint32_t CPUINFO_ABI cpuinfo_get_cores_count(void);

#endif /* CPUINFO_H */
