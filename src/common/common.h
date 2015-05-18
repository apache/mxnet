#pragma once
#include <vector>
typedef float FloatT;
typedef std::vector<int> Scale;
inline size_t Capacity(const Scale & s) {
  size_t r = 1;
  for (auto ss : s)
    r *= ss;
  return r * sizeof(FloatT);
}