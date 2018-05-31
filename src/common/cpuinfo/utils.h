#pragma once

#include <stdint.h>


inline static uint32_t bit_length(uint32_t n) {
	const uint32_t n_minus_1 = n - 1;
	if (n_minus_1 == 0) {
		return 0;
	} else {
		#ifdef _MSC_VER
			unsigned long bsr;
			_BitScanReverse(&bsr, n_minus_1);
			return bsr + 1;
		#else
			return 32 - __builtin_clz(n_minus_1);
		#endif
	}
}
