#if defined(__linux__)

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "../../cpuinfo.h"
#include "../api.h"
#include "./api.h"
#include "../log.h"


static inline uint32_t bit_mask(uint32_t bits) {
	return (UINT32_C(1) << bits) - UINT32_C(1);
}

static inline bool bitmask_all(uint32_t bitfield, uint32_t mask) {
	return (bitfield & mask) == mask;
}

static inline uint32_t min(uint32_t a, uint32_t b) {
	return a < b ? a : b;
}

static inline int cmp(uint32_t a, uint32_t b) {
	return (a > b) - (a < b);
}

static int cmp_x86_linux_processor(const void* ptr_a, const void* ptr_b) {
	const struct cpuinfo_x86_linux_processor* processor_a = (const struct cpuinfo_x86_linux_processor*) ptr_a;
	const struct cpuinfo_x86_linux_processor* processor_b = (const struct cpuinfo_x86_linux_processor*) ptr_b;

	/* Move usable processors towards the start of the array */
	const bool usable_a = bitmask_all(processor_a->flags, CPUINFO_LINUX_MASK_USABLE);
	const bool usable_b = bitmask_all(processor_b->flags, CPUINFO_LINUX_MASK_USABLE);
	if (usable_a != usable_b) {
		return (int) usable_b - (int) usable_a;
	}

	/* Compare based on APIC ID (i.e. processor 0 < processor 1) */
	const uint32_t id_a = processor_a->apic_id;
	const uint32_t id_b = processor_b->apic_id;
	return cmp(id_a, id_b);
}

static void cpuinfo_x86_count_objects(
	uint32_t linux_processors_count,
	const cpuinfo_x86_linux_processor* linux_processors,
	const struct cpuinfo_x86_processor processor[1],
	uint32_t llc_apic_bits,
	uint32_t cores_count_ptr[1],
	uint32_t clusters_count_ptr[1],
	uint32_t packages_count_ptr[1],
	uint32_t l1i_count_ptr[1],
	uint32_t l1d_count_ptr[1],
	uint32_t l2_count_ptr[1],
	uint32_t l3_count_ptr[1],
	uint32_t l4_count_ptr[1])
{
	const uint32_t core_apic_mask =
		~(bit_mask(processor->topology.thread_bits_length) << processor->topology.thread_bits_offset);
	const uint32_t package_apic_mask =
		core_apic_mask & ~(bit_mask(processor->topology.core_bits_length) << processor->topology.core_bits_offset);
	const uint32_t llc_apic_mask = ~bit_mask(llc_apic_bits);
	const uint32_t cluster_apic_mask = package_apic_mask | llc_apic_mask;

	uint32_t cores_count = 0, clusters_count = 0, packages_count = 0;
	uint32_t l1i_count = 0, l1d_count = 0, l2_count = 0, l3_count = 0, l4_count = 0;
	uint32_t last_core_id = UINT32_MAX, last_cluster_id = UINT32_MAX, last_package_id = UINT32_MAX;
	uint32_t last_l1i_id = UINT32_MAX, last_l1d_id = UINT32_MAX;
	uint32_t last_l2_id = UINT32_MAX, last_l3_id = UINT32_MAX, last_l4_id = UINT32_MAX;
	for (uint32_t i = 0; i < linux_processors_count; i++) {
		if (bitmask_all(linux_processors[i].flags, CPUINFO_LINUX_MASK_USABLE)) {
			const uint32_t apic_id = linux_processors[i].apic_id;
			cpuinfo_log_debug("APID ID %"PRIu32": system processor %"PRIu32, apic_id, linux_processors[i].linux_id);

			/* All bits of APIC ID except thread ID mask */
			const uint32_t core_id = apic_id & core_apic_mask;
			if (core_id != last_core_id) {
				last_core_id = core_id;
				cores_count++;
			}
		}
	}
	*cores_count_ptr = cores_count;
	*clusters_count_ptr = clusters_count;
	*packages_count_ptr = packages_count;
	*l1i_count_ptr = l1i_count;
	*l1d_count_ptr = l1d_count;
	*l2_count_ptr  = l2_count;
	*l3_count_ptr  = l3_count;
	*l4_count_ptr  = l4_count;
}

void cpuinfo_x86_linux_init(void) {
	struct cpuinfo_x86_linux_processor* x86_linux_processors = NULL;
	struct cpuinfo_core* cores = NULL;

	const uint32_t max_processors_count = cpuinfo_linux_get_max_processors_count();
	cpuinfo_log_debug("system maximum processors count: %"PRIu32, max_processors_count);

	const uint32_t max_possible_processors_count = 1 +
		cpuinfo_linux_get_max_possible_processor(max_processors_count);
	cpuinfo_log_debug("maximum possible processors count: %"PRIu32, max_possible_processors_count);
	const uint32_t max_present_processors_count = 1 +
		cpuinfo_linux_get_max_present_processor(max_processors_count);
	cpuinfo_log_debug("maximum present processors count: %"PRIu32, max_present_processors_count);

	const uint32_t x86_linux_processors_count = min(max_possible_processors_count, max_present_processors_count);
	x86_linux_processors = (cpuinfo_x86_linux_processor*) calloc(x86_linux_processors_count, sizeof(struct cpuinfo_x86_linux_processor));
	if (x86_linux_processors == NULL) {
		cpuinfo_log_error(
			"failed to allocate %zu bytes for descriptions of %"PRIu32" x86 logical processors",
			x86_linux_processors_count * sizeof(struct cpuinfo_x86_linux_processor),
			x86_linux_processors_count);
		return;
	}

	cpuinfo_linux_detect_possible_processors(
		x86_linux_processors_count, &x86_linux_processors->flags,
		sizeof(struct cpuinfo_x86_linux_processor),
		CPUINFO_LINUX_FLAG_POSSIBLE);

	cpuinfo_linux_detect_present_processors(
		x86_linux_processors_count, &x86_linux_processors->flags,
		sizeof(struct cpuinfo_x86_linux_processor),
		CPUINFO_LINUX_FLAG_PRESENT);

	if (!cpuinfo_x86_linux_parse_proc_cpuinfo(x86_linux_processors_count, x86_linux_processors)) {
		cpuinfo_log_error("failed to parse processor information from /proc/cpuinfo");
		return;
	}

	struct cpuinfo_x86_processor x86_processor;
	memset(&x86_processor, 0, sizeof(x86_processor));
	cpuinfo_x86_init_processor(&x86_processor);
	char brand_string[48];

	qsort(x86_linux_processors, x86_linux_processors_count, sizeof(struct cpuinfo_x86_linux_processor),
		cmp_x86_linux_processor);

	uint32_t llc_apic_bits = 0;
	uint32_t packages_count = 0, clusters_count = 0, cores_count = 0;
	uint32_t l1i_count = 0, l1d_count = 0, l2_count = 0, l3_count = 0, l4_count = 0;
	cpuinfo_x86_count_objects(x86_linux_processors_count, x86_linux_processors, &x86_processor, llc_apic_bits,
		&cores_count, &clusters_count, &packages_count, &l1i_count, &l1d_count, &l2_count, &l3_count, &l4_count);

	cpuinfo_log_debug("detected %"PRIu32" cores", cores_count);

	cpuinfo_cores_count = cores_count;

	__sync_synchronize();

	cpuinfo_is_initialized = true;

cleanup:
	free(cores);
}

#endif